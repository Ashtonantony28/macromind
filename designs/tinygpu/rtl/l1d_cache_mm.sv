`default_nettype none
`timescale 1ns/1ns

// MacroMind: banked L1-D cache on single-port SRAM macros
// (replaces upstream src/l1d_cache.sv in the macro variant; same ports).
//
// Upstream: one direct-mapped, write-through, 1-word-line cache with flop
// arrays and NUM_CONSUMERS concurrent lookup ports.
// Here: BANKS address-interleaved banks (bank = addr % BANKS), each one
// mm_sram of LINES/BANKS words holding {tag, data} (fakeram45_256x34 for
// 1024 lines / 4 banks / 12-bit addr / 32-bit data). Valid bits stay in flops.
// Each bank serves one lane per cycle (round-robin), so a lookup is
//   IDLE -> RREQ (wait for bank) -> RLOOK (SRAM data) -> hit: ready / miss: MISS
// and a miss installs through FILL (another bank slot) unless MeDiC bypasses;
// a write-through completes with INV (a bank slot that clears the valid bit).
// Writes are write-through + write-INVALIDATE (upstream: update-on-hit). Both
// keep the cache coherent with memory for a single GPU, so kernel results are
// unchanged; hit/miss telemetry and cycle counts differ.
module l1d_cache_mm #(
    parameter NUM_CONSUMERS    = 8,
    parameter ADDR_BITS        = 8,
    parameter DATA_BITS        = 8,
    parameter LINES            = 16,
    parameter BANKS            = 1,
    parameter MEDIC            = 0,
    parameter MEDIC_THRESHOLD  = 2
) (
    input  wire                              clk,
    input  wire                              reset,
    input  wire                              telemetry_reset,

    input  wire [NUM_CONSUMERS-1:0]          consumer_read_valid,
    input  wire [ADDR_BITS-1:0]              consumer_read_address  [NUM_CONSUMERS-1:0],
    output reg  [NUM_CONSUMERS-1:0]          consumer_read_ready,
    output reg  [DATA_BITS-1:0]              consumer_read_data     [NUM_CONSUMERS-1:0],
    input  wire [NUM_CONSUMERS-1:0]          consumer_write_valid,
    input  wire [ADDR_BITS-1:0]              consumer_write_address [NUM_CONSUMERS-1:0],
    input  wire [DATA_BITS-1:0]              consumer_write_data    [NUM_CONSUMERS-1:0],
    output reg  [NUM_CONSUMERS-1:0]          consumer_write_ready,

    output reg  [NUM_CONSUMERS-1:0]          mem_read_valid,
    output reg  [ADDR_BITS-1:0]              mem_read_address       [NUM_CONSUMERS-1:0],
    input  wire [NUM_CONSUMERS-1:0]          mem_read_ready,
    input  wire [DATA_BITS-1:0]              mem_read_data          [NUM_CONSUMERS-1:0],
    output reg  [NUM_CONSUMERS-1:0]          mem_write_valid,
    output reg  [ADDR_BITS-1:0]              mem_write_address      [NUM_CONSUMERS-1:0],
    output reg  [DATA_BITS-1:0]              mem_write_data         [NUM_CONSUMERS-1:0],
    input  wire [NUM_CONSUMERS-1:0]          mem_write_ready,

    output wire [31:0]                       cache_hits_total,
    output wire [31:0]                       cache_misses_total,
    output wire [NUM_CONSUMERS-1:0]          warp_low_utility,
    output wire [NUM_CONSUMERS-1:0]          warp_high_utility
);
    localparam BB       = (BANKS <= 1) ? 0 : $clog2(BANKS);
    localparam LPB      = LINES / BANKS;                       // lines per bank
    localparam IB       = (LPB <= 1) ? 1 : $clog2(LPB);
    localparam LB       = BB + IB;                             // line-number bits
    localparam TAG_BITS = (ADDR_BITS > LB) ? (ADDR_BITS - LB) : 1;
    localparam SW       = TAG_BITS + DATA_BITS;                // SRAM word
    localparam CW       = (NUM_CONSUMERS <= 1) ? 1 : $clog2(NUM_CONSUMERS);
    localparam BW       = (BANKS <= 1) ? 1 : BB;

    localparam [2:0] L_IDLE = 3'd0, L_RREQ = 3'd1, L_RLOOK = 3'd2, L_MISS = 3'd3,
                     L_FILL = 3'd4, L_WT = 3'd5, L_INV = 3'd6;

    function automatic [BW-1:0] bank_of(input [ADDR_BITS-1:0] a);
        bank_of = (BANKS <= 1) ? {BW{1'b0}} : a[BW-1:0];
    endfunction
    function automatic [IB-1:0] idx_of(input [ADDR_BITS-1:0] a);
        idx_of = a[BB +: IB];
    endfunction
    function automatic [LB-1:0] line_of(input [ADDR_BITS-1:0] a);
        line_of = a[LB-1:0];
    endfunction
    function automatic [TAG_BITS-1:0] tag_of(input [ADDR_BITS-1:0] a);
        tag_of = a >> LB;
    endfunction

    reg [2:0]           state [NUM_CONSUMERS-1:0];
    reg [ADDR_BITS-1:0] laddr [NUM_CONSUMERS-1:0];
    reg [DATA_BITS-1:0] ldata [NUM_CONSUMERS-1:0];
    // Valid bits, one vector per bank. Only the lane granted a bank in a cycle
    // reads or writes that bank's valid bits (FILL sets, INV clears, RREQ reads
    // into bank_vld alongside the SRAM read), so each bank has one decoder and one
    // read mux instead of one per lane.
    reg [LPB-1:0]       vld      [BANKS-1:0];
    reg [BANKS-1:0]     bank_vld;
    reg [3:0]           hit_window [NUM_CONSUMERS-1:0];
    reg [31:0]          hits_total, misses_total;
    assign cache_hits_total   = hits_total;
    assign cache_misses_total = misses_total;

    genvar lu;
    generate
        for (lu = 0; lu < NUM_CONSUMERS; lu = lu + 1) begin : util
            wire [2:0] h = hit_window[lu][3] + hit_window[lu][2] + hit_window[lu][1] + hit_window[lu][0];
            assign warp_low_utility[lu]  = (h < MEDIC_THRESHOLD);
            assign warp_high_utility[lu] = (h >= MEDIC_THRESHOLD);
        end
    endgenerate

    // ---- per-bank round-robin arbitration (combinational) ----
    reg [NUM_CONSUMERS-1:0] lane_gnt;
    reg [BANKS-1:0]         bank_any;
    reg [CW-1:0]            bank_gidx [BANKS-1:0];
    reg [CW-1:0]            rr_ptr    [BANKS-1:0];
    reg                     found_hi, found_lo;
    reg [CW-1:0]            pick_hi, pick_lo;
    integer b, i;
    always @(*) begin
        lane_gnt = {NUM_CONSUMERS{1'b0}};
        for (b = 0; b < BANKS; b = b + 1) begin
            found_hi = 1'b0; found_lo = 1'b0;
            pick_hi  = {CW{1'b0}}; pick_lo = {CW{1'b0}};
            for (i = NUM_CONSUMERS - 1; i >= 0; i = i - 1) begin
                if ((state[i] == L_RREQ || state[i] == L_FILL || state[i] == L_INV) && bank_of(laddr[i]) == b) begin
                    if (i >= rr_ptr[b]) begin found_hi = 1'b1; pick_hi = i[CW-1:0]; end
                    found_lo = 1'b1; pick_lo = i[CW-1:0];
                end
            end
            bank_any[b]  = found_hi | found_lo;
            bank_gidx[b] = found_hi ? pick_hi : pick_lo;
            if (bank_any[b]) lane_gnt[bank_gidx[b]] = 1'b1;
        end
    end

    // ---- SRAM banks ----
    wire [SW-1:0] bank_q [BANKS-1:0];
    genvar gb;
    generate
        for (gb = 0; gb < BANKS; gb = gb + 1) begin : bank
            wire [CW-1:0] g    = bank_gidx[gb];
            wire [2:0]    gst  = state[g];
            wire [IB-1:0] gidx = idx_of(laddr[g]);
            wire          fill = gst == L_FILL;
            always @(posedge clk) begin
                if (reset) begin
                    vld[gb]      <= {LPB{1'b0}};
                    bank_vld[gb] <= 1'b0;
                end else if (bank_any[gb]) begin
                    if (gst == L_FILL) vld[gb][gidx] <= 1'b1;
                    if (gst == L_INV)  vld[gb][gidx] <= 1'b0;
                    if (gst == L_RREQ) bank_vld[gb] <= vld[gb][gidx];
                end
            end
            mm_sram #(.DEPTH(LPB), .WIDTH(SW)) u_sram (
                .clk(clk), .ce(bank_any[gb] && gst != L_INV && !reset), .we(fill),
                .addr(gidx), .wdata({tag_of(laddr[g]), ldata[g]}),
                .rdata(bank_q[gb])
            );
        end
    endgenerate

    integer hit_inc, miss_inc;
    reg [SW-1:0] q;
    reg          hit;

    always @(posedge clk) begin
        if (reset) begin
            for (i = 0; i < NUM_CONSUMERS; i = i + 1) begin
                state[i]                <= L_IDLE;
                laddr[i]                <= {ADDR_BITS{1'b0}};
                ldata[i]                <= {DATA_BITS{1'b0}};
                consumer_read_ready[i]  <= 1'b0;
                consumer_read_data[i]   <= {DATA_BITS{1'b0}};
                consumer_write_ready[i] <= 1'b0;
                mem_read_valid[i]       <= 1'b0;
                mem_read_address[i]     <= {ADDR_BITS{1'b0}};
                mem_write_valid[i]      <= 1'b0;
                mem_write_address[i]    <= {ADDR_BITS{1'b0}};
                mem_write_data[i]       <= {DATA_BITS{1'b0}};
                hit_window[i]           <= 4'b1111;
            end
            for (b = 0; b < BANKS; b = b + 1) rr_ptr[b] <= {CW{1'b0}};
            hits_total   <= 32'b0;
            misses_total <= 32'b0;
        end else begin
            for (b = 0; b < BANKS; b = b + 1)
                if (bank_any[b]) rr_ptr[b] <= bank_gidx[b] + 1'b1;
            hit_inc  = 0;
            miss_inc = 0;
            for (i = 0; i < NUM_CONSUMERS; i = i + 1) begin
                // ready drops as soon as the consumer drops valid, in any state
                if (consumer_read_ready[i] && !consumer_read_valid[i])   consumer_read_ready[i]  <= 1'b0;
                if (consumer_write_ready[i] && !consumer_write_valid[i]) consumer_write_ready[i] <= 1'b0;
                case (state[i])
                    L_IDLE: begin
                        if (consumer_read_valid[i] && !consumer_read_ready[i]) begin
                            laddr[i] <= consumer_read_address[i];
                            state[i] <= L_RREQ;
                        end else if (consumer_write_valid[i] && !consumer_write_ready[i]) begin
                            mem_write_valid[i]   <= 1'b1;
                            mem_write_address[i] <= consumer_write_address[i];
                            mem_write_data[i]    <= consumer_write_data[i];
                            state[i]             <= L_WT;
                        end
                    end
                    L_RREQ: if (lane_gnt[i]) state[i] <= L_RLOOK;
                    L_RLOOK: begin
                        q   = bank_q[bank_of(laddr[i])];
                        hit = bank_vld[bank_of(laddr[i])] && q[SW-1:DATA_BITS] == tag_of(laddr[i]);
                        if (hit) begin
                            consumer_read_data[i]  <= q[DATA_BITS-1:0];
                            consumer_read_ready[i] <= 1'b1;
                            hit_inc                = hit_inc + 1;
                            hit_window[i]          <= {hit_window[i][2:0], 1'b1};
                            state[i]               <= L_IDLE;
                        end else begin
                            mem_read_valid[i]   <= 1'b1;
                            mem_read_address[i] <= laddr[i];
                            miss_inc            = miss_inc + 1;
                            hit_window[i]       <= {hit_window[i][2:0], 1'b0};
                            state[i]            <= L_MISS;
                        end
                    end
                    L_MISS: if (mem_read_ready[i]) begin
                        mem_read_valid[i]      <= 1'b0;
                        consumer_read_data[i]  <= mem_read_data[i];
                        consumer_read_ready[i] <= 1'b1;
                        ldata[i]               <= mem_read_data[i];
                        state[i]               <= (MEDIC && warp_low_utility[i]) ? L_IDLE : L_FILL;
                    end
                    L_FILL: if (lane_gnt[i]) state[i] <= L_IDLE;
                    L_INV:  if (lane_gnt[i]) state[i] <= L_IDLE;
                    L_WT: if (mem_write_ready[i]) begin
                        mem_write_valid[i]      <= 1'b0;
                        consumer_write_ready[i] <= 1'b1;
                        laddr[i]                <= mem_write_address[i];
                        state[i]                <= L_INV;            // write-invalidate
                    end
                    default: state[i] <= L_IDLE;
                endcase
            end
            if (telemetry_reset) begin
                hits_total   <= 32'b0;
                misses_total <= 32'b0;
                for (i = 0; i < NUM_CONSUMERS; i = i + 1) hit_window[i] <= 4'b1111;
            end else begin
                hits_total   <= hits_total   + hit_inc[31:0];
                misses_total <= misses_total + miss_inc[31:0];
            end
        end
    end
endmodule
