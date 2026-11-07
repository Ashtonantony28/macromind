`default_nettype none
`timescale 1ns/1ns
// DERIVED by designs/tinygpu/derive_rtl.py from upstream src/lsu.sv
// (Didactic_GPU_Simulator, commit in designs/tinygpu/UPSTREAM_PIN). Do not edit by hand.
// MacroMind change: module renamed lsu_mm; new input smem_ready. The shared-memory
//   read/write WAITING states now wait for smem_ready (the SRAM-macro shared memory
//   serves one lane per cycle) instead of completing unconditionally after one cycle.

// Load-Store Unit
// One per thread. Async global memory request → wait → done handshake.
// Also handles per-block shared-memory access (Phase 7) via a separate port.
// At decode time at most one of {global, shared} is enabled — they share the FSM.
module lsu_mm #(
    parameter WORD_BITS = 8,
    parameter ADDR_BITS = 8
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        enable,
    input  wire [2:0]  core_state,
    input  wire        decoded_mem_read_enable,
    input  wire        decoded_mem_write_enable,
    input  wire        decoded_smem_read_enable,
    input  wire        decoded_smem_write_enable,
    input  wire [WORD_BITS-1:0] rs,
    input  wire [WORD_BITS-1:0] rt,

    // Global memory
    output reg         mem_read_valid,
    output reg  [ADDR_BITS-1:0] mem_read_address,
    input  wire        mem_read_ready,
    input  wire [WORD_BITS-1:0] mem_read_data,
    output reg         mem_write_valid,
    output reg  [ADDR_BITS-1:0] mem_write_address,
    output reg  [WORD_BITS-1:0] mem_write_data,
    input  wire        mem_write_ready,

    // Shared memory (single-cycle)
    output reg         smem_read_enable,
    output reg  [7:0]  smem_read_address,
    input  wire [WORD_BITS-1:0] smem_read_data,
    input  wire        smem_ready,
    output reg         smem_write_enable,
    output reg  [7:0]  smem_write_address,
    output reg  [WORD_BITS-1:0] smem_write_data,

    output reg  [1:0]  lsu_state,
    output reg  [WORD_BITS-1:0] lsu_out
);
    localparam [1:0] IDLE = 2'b00, REQUESTING = 2'b01, WAITING = 2'b10, DONE = 2'b11;
    localparam [2:0] CORE_REQUEST = 3'b011, CORE_UPDATE = 3'b110;

    always @(posedge clk) begin
        if (reset) begin
            lsu_state          <= IDLE;
            lsu_out            <= 0;
            mem_read_valid     <= 0;
            mem_read_address   <= 0;
            mem_write_valid    <= 0;
            mem_write_address  <= 0;
            mem_write_data     <= 0;
            smem_read_enable   <= 0;
            smem_read_address  <= 0;
            smem_write_enable  <= 0;
            smem_write_address <= 0;
            smem_write_data    <= 0;
        end else if (enable) begin
            // GLOBAL READ
            if (decoded_mem_read_enable) begin
                case (lsu_state)
                    IDLE:       if (core_state == CORE_REQUEST) lsu_state <= REQUESTING;
                    REQUESTING: begin
                        mem_read_valid   <= 1;
                        mem_read_address <= rs[ADDR_BITS-1:0];
                        lsu_state        <= WAITING;
                    end
                    WAITING: if (mem_read_ready) begin
                        mem_read_valid <= 0;
                        lsu_out        <= mem_read_data;
                        lsu_state      <= DONE;
                    end
                    DONE: if (core_state == CORE_UPDATE) lsu_state <= IDLE;
                endcase
            end
            // GLOBAL WRITE
            if (decoded_mem_write_enable) begin
                case (lsu_state)
                    IDLE:       if (core_state == CORE_REQUEST) lsu_state <= REQUESTING;
                    REQUESTING: begin
                        mem_write_valid   <= 1;
                        mem_write_address <= rs[ADDR_BITS-1:0];
                        mem_write_data    <= rt;
                        lsu_state         <= WAITING;
                    end
                    WAITING: if (mem_write_ready) begin
                        mem_write_valid <= 0;
                        lsu_state       <= DONE;
                    end
                    DONE: if (core_state == CORE_UPDATE) lsu_state <= IDLE;
                endcase
            end
            // SHARED READ (single-cycle SRAM, registered output)
            // Mirror the global pattern: idle→requesting first, then sample
            // rs/rt in REQUESTING so the register file has had a cycle to update.
            if (decoded_smem_read_enable) begin
                case (lsu_state)
                    IDLE: if (core_state == CORE_REQUEST) lsu_state <= REQUESTING;
                    REQUESTING: begin
                        smem_read_enable  <= 1;
                        smem_read_address <= rs[7:0];
                        lsu_state         <= WAITING;
                    end
                    WAITING: if (smem_ready) begin
                        smem_read_enable <= 0;
                        lsu_out          <= smem_read_data;
                        lsu_state        <= DONE;
                    end
                    DONE: if (core_state == CORE_UPDATE) lsu_state <= IDLE;
                endcase
            end
            // SHARED WRITE
            if (decoded_smem_write_enable) begin
                case (lsu_state)
                    IDLE: if (core_state == CORE_REQUEST) lsu_state <= REQUESTING;
                    REQUESTING: begin
                        smem_write_enable  <= 1;
                        smem_write_address <= rs[7:0];
                        smem_write_data    <= rt;
                        lsu_state          <= WAITING;
                    end
                    WAITING: if (smem_ready) begin
                        smem_write_enable <= 0;
                        lsu_state         <= DONE;
                    end
                    DONE: if (core_state == CORE_UPDATE) lsu_state <= IDLE;
                endcase
            end
        end
    end
endmodule
