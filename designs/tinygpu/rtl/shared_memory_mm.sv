`default_nettype none
`timescale 1ns/1ns

// MacroMind: per-block shared memory on one single-port SRAM macro
// (replaces upstream src/shared_memory.sv in the macro variant).
//
// Upstream: SIZE x DATA_BITS flop array, THREADS_PER_BLOCK combinational read
// ports and THREADS_PER_BLOCK write ports, cleared on (per-block) reset.
// Here: one mm_sram (fakeram45_64x32 for SIZE=64, DATA_BITS=32) serving one
// lane per cycle. A lane holds lane_read_enable / lane_write_enable (the
// lsu_mm keeps them high in WAITING) until lane_ready pulses for one cycle;
// read data is valid on lane_read_data in that cycle. Lanes are served in
// ascending order, so same-address writes from several lanes in one STS end
// with the highest lane's value, as upstream. Reset cannot clear an SRAM, so a
// SIZE-bit valid vector (cleared on reset) makes unwritten words read as 0,
// exactly like the upstream reset-to-zero.
module shared_memory_mm #(
    parameter THREADS_PER_BLOCK = 4,
    parameter SIZE              = 64,
    parameter ADDR_BITS         = 8,
    parameter DATA_BITS         = 8
) (
    input  wire                          clk,
    input  wire                          reset,
    input  wire [THREADS_PER_BLOCK-1:0]  lane_read_enable,
    input  wire [ADDR_BITS-1:0]          lane_read_addr  [THREADS_PER_BLOCK-1:0],
    output wire [DATA_BITS-1:0]          lane_read_data  [THREADS_PER_BLOCK-1:0],
    input  wire [THREADS_PER_BLOCK-1:0]  lane_write_enable,
    input  wire [ADDR_BITS-1:0]          lane_write_addr [THREADS_PER_BLOCK-1:0],
    input  wire [DATA_BITS-1:0]          lane_write_data [THREADS_PER_BLOCK-1:0],
    output reg  [THREADS_PER_BLOCK-1:0]  lane_ready
);
    localparam IDX_BITS = (SIZE <= 1) ? 1 : $clog2(SIZE);

    wire [THREADS_PER_BLOCK-1:0] pending = (lane_read_enable | lane_write_enable) & ~lane_ready;

    // Lowest pending lane wins.
    reg                          gnt_any;
    reg [THREADS_PER_BLOCK-1:0]  gnt;
    reg                          gnt_we;
    reg [IDX_BITS-1:0]           gnt_addr;
    reg [DATA_BITS-1:0]          gnt_wdata;
    integer k;
    always @(*) begin
        gnt_any   = 1'b0;
        gnt       = {THREADS_PER_BLOCK{1'b0}};
        gnt_we    = 1'b0;
        gnt_addr  = {IDX_BITS{1'b0}};
        gnt_wdata = {DATA_BITS{1'b0}};
        for (k = THREADS_PER_BLOCK - 1; k >= 0; k = k - 1) begin
            if (pending[k]) begin
                gnt_any   = 1'b1;
                gnt       = {{(THREADS_PER_BLOCK-1){1'b0}}, 1'b1} << k;
                gnt_we    = lane_write_enable[k];
                gnt_addr  = lane_write_enable[k] ? lane_write_addr[k][IDX_BITS-1:0]
                                                 : lane_read_addr[k][IDX_BITS-1:0];
                gnt_wdata = lane_write_data[k];
            end
        end
    end

    wire [DATA_BITS-1:0] sram_q;
    mm_sram #(.DEPTH(SIZE), .WIDTH(DATA_BITS)) u_sram (
        .clk(clk), .ce(gnt_any && !reset), .we(gnt_we), .addr(gnt_addr),
        .wdata(gnt_wdata), .rdata(sram_q)
    );

    reg [SIZE-1:0] written;
    reg            rd_written;
    always @(posedge clk) begin
        if (reset) begin
            written    <= {SIZE{1'b0}};
            lane_ready <= {THREADS_PER_BLOCK{1'b0}};
            rd_written <= 1'b0;
        end else begin
            lane_ready <= gnt;
            rd_written <= written[gnt_addr];
            if (gnt_any && gnt_we) written[gnt_addr] <= 1'b1;
        end
    end

    wire [DATA_BITS-1:0] q = rd_written ? sram_q : {DATA_BITS{1'b0}};
    genvar g;
    generate
        for (g = 0; g < THREADS_PER_BLOCK; g = g + 1) begin : rports
            assign lane_read_data[g] = q;
        end
    endgenerate
endmodule
