`default_nettype none
`timescale 1ns/1ns
// MacroMind: copy of upstream src/fetcher.sv (module renamed fetcher_mm). The
// ICACHE_LINES=0 path is unchanged; the I-cache path stores {tag, instruction}
// in an SRAM macro (mm_sram) instead of flop arrays.

// Instruction Fetcher with optional I-cache (Phase 3).
// ICACHE_LINES = 0 → cache bypassed (behaves like upstream).
// telemetry_reset drives a separate global domain so the per-block `reset`
// doesn't wipe icache_hits / icache_misses.
module fetcher_mm #(
    parameter PROGRAM_MEM_ADDR_BITS = 8,
    parameter PROGRAM_MEM_DATA_BITS = 16,
    parameter ICACHE_LINES = 0
) (
    input  wire                              clk,
    input  wire                              reset,
    input  wire                              telemetry_reset,
    input  wire [2:0]                        core_state,
    input  wire [PROGRAM_MEM_ADDR_BITS-1:0]  current_pc,

    output reg                               mem_read_valid,
    output reg  [PROGRAM_MEM_ADDR_BITS-1:0]  mem_read_address,
    input  wire                              mem_read_ready,
    input  wire [PROGRAM_MEM_DATA_BITS-1:0]  mem_read_data,

    output reg  [2:0]                        fetcher_state,
    output reg  [PROGRAM_MEM_DATA_BITS-1:0]  instruction,

    output reg  [31:0]                       icache_hits,
    output reg  [31:0]                       icache_misses
);
    localparam [2:0] IDLE = 3'b000, FETCHING = 3'b001, FETCHED = 3'b010;
    localparam [2:0] CORE_FETCH = 3'b001, CORE_DECODE = 3'b010, CORE_REQUEST = 3'b011;

    // 1-cycle pulse: a fetch was initiated this cycle.
    reg fetch_initiated;
    reg fetch_was_hit;

    always @(posedge clk) begin
        if (telemetry_reset) begin
            icache_hits   <= 0;
            icache_misses <= 0;
        end else if (fetch_initiated) begin
            if (fetch_was_hit) icache_hits   <= icache_hits   + 1;
            else               icache_misses <= icache_misses + 1;
        end
    end

    generate
        if (ICACHE_LINES == 0) begin : no_icache
            always @(posedge clk) begin
                if (reset) begin
                    fetcher_state    <= IDLE;
                    mem_read_valid   <= 0;
                    mem_read_address <= 0;
                    instruction      <= {PROGRAM_MEM_DATA_BITS{1'b0}};
                    fetch_initiated  <= 1'b0;
                    fetch_was_hit    <= 1'b0;
                end else begin
                    fetch_initiated <= 1'b0;
                    case (fetcher_state)
                        IDLE: if (core_state == CORE_FETCH) begin
                            fetcher_state    <= FETCHING;
                            mem_read_valid   <= 1;
                            mem_read_address <= current_pc;
                            fetch_initiated  <= 1'b1;
                            fetch_was_hit    <= 1'b0;
                        end
                        FETCHING: if (mem_read_ready) begin
                            fetcher_state  <= FETCHED;
                            instruction    <= mem_read_data;
                            mem_read_valid <= 0;
                        end
                        // Move out of FETCHED when the scheduler advances past FETCH.
                        // In pipelined mode the scheduler skips DECODE → REQUEST.
                        FETCHED: if (core_state == CORE_DECODE || core_state == CORE_REQUEST) fetcher_state <= IDLE;
                        default: ;
                    endcase
                end
            end
        end else begin : icache
            // MacroMind: tag+instruction array in one mm_sram (fakeram45_64x21 for
            // 64 lines, 8-bit PC, 16-bit instructions); valid bits stay in flops.
            // The SRAM read takes one cycle, so a fetch is IDLE -> LOOKUP -> (hit)
            // FETCHED or (miss) FETCHING -> FETCHED: one extra cycle per fetch.
            localparam INDEX_BITS = (ICACHE_LINES <= 1) ? 1 : $clog2(ICACHE_LINES);
            localparam TAG_BITS   = (PROGRAM_MEM_ADDR_BITS > INDEX_BITS)
                                    ? (PROGRAM_MEM_ADDR_BITS - INDEX_BITS) : 1;
            localparam [2:0] LOOKUP = 3'b011;   // internal; the scheduler only tests FETCHED

            reg  [ICACHE_LINES-1:0]                   valid_arr;
            wire [INDEX_BITS-1:0]                     idx = current_pc[INDEX_BITS-1:0];
            wire [TAG_BITS-1:0]                       tag = current_pc[PROGRAM_MEM_ADDR_BITS-1:INDEX_BITS];
            wire [TAG_BITS+PROGRAM_MEM_DATA_BITS-1:0] line_q;
            wire                                      hit = valid_arr[idx] && (line_q[TAG_BITS+PROGRAM_MEM_DATA_BITS-1:PROGRAM_MEM_DATA_BITS] == tag);
            wire do_read = !reset && fetcher_state == IDLE && core_state == CORE_FETCH;
            wire do_fill = !reset && fetcher_state == FETCHING && mem_read_ready;

            mm_sram #(.DEPTH(ICACHE_LINES), .WIDTH(TAG_BITS + PROGRAM_MEM_DATA_BITS)) u_lines (
                .clk(clk), .ce(do_read || do_fill), .we(do_fill), .addr(idx),
                .wdata({tag, mem_read_data}), .rdata(line_q)
            );

            // Valid bits: wiped on telemetry_reset (fresh kernel launch), as the
            // upstream comment intends (see docs/tinygpu-macro-variant.md).
            always @(posedge clk) begin
                if (telemetry_reset) valid_arr <= {ICACHE_LINES{1'b0}};
                else if (do_fill)    valid_arr[idx] <= 1'b1;
            end

            always @(posedge clk) begin
                if (reset) begin
                    fetcher_state    <= IDLE;
                    mem_read_valid   <= 0;
                    mem_read_address <= 0;
                    instruction      <= {PROGRAM_MEM_DATA_BITS{1'b0}};
                    fetch_initiated  <= 1'b0;
                    fetch_was_hit    <= 1'b0;
                end else begin
                    fetch_initiated <= 1'b0;
                    case (fetcher_state)
                        IDLE: if (core_state == CORE_FETCH) fetcher_state <= LOOKUP;
                        LOOKUP: begin
                            fetch_initiated <= 1'b1;
                            fetch_was_hit   <= hit;
                            if (hit) begin
                                instruction   <= line_q[PROGRAM_MEM_DATA_BITS-1:0];
                                fetcher_state <= FETCHED;
                            end else begin
                                fetcher_state    <= FETCHING;
                                mem_read_valid   <= 1;
                                mem_read_address <= current_pc;
                            end
                        end
                        FETCHING: if (mem_read_ready) begin
                            fetcher_state  <= FETCHED;
                            instruction    <= mem_read_data;
                            mem_read_valid <= 0;
                        end
                        // Move out of FETCHED when the scheduler advances past FETCH.
                        // In pipelined mode the scheduler skips DECODE → REQUEST.
                        FETCHED: if (core_state == CORE_DECODE || core_state == CORE_REQUEST) fetcher_state <= IDLE;
                        default: ;
                    endcase
                end
            end
        end
    endgenerate
endmodule
