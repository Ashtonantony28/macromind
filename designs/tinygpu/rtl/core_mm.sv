`default_nettype none
`timescale 1ns/1ns
// DERIVED by designs/tinygpu/derive_rtl.py from upstream src/core.sv
// (Didactic_GPU_Simulator, commit in designs/tinygpu/UPSTREAM_PIN). Do not edit by hand.
// MacroMind change: module renamed core_mm; instantiates fetcher_mm (SRAM I-cache),
//   shared_memory_mm (SRAM shared memory, one lane per cycle) and lsu_mm (waits for
//   smem_ready). Everything else is upstream.

// Compute Core
// One block at a time. One fetcher/decoder/scheduler shared across THREADS_PER_BLOCK threads, split into
// WARPS_PER_CORE warps of LANES consecutive threads (ADR-012, ADR-014). Each thread has its own registers,
// NZP/PC and LSU; the LANES ALUs are time-shared by the warps. Per-block shared memory (Phase 7) is here.
// SIMT_STACK=1 gives each warp a reconvergence stack whose top mask gates its lanes (ADR-011, ADR-015).
module core_mm #(
    parameter DATA_MEM_ADDR_BITS    = 8,
    parameter DATA_MEM_DATA_BITS    = 8,
    parameter WORD_BITS             = 8,
    parameter PROGRAM_MEM_ADDR_BITS = 8,
    parameter PROGRAM_MEM_DATA_BITS = 16,
    parameter THREADS_PER_BLOCK     = 4,
    parameter ICACHE_LINES          = 0,
    parameter SMEM_SIZE             = 64,
    parameter PIPELINED             = 0,
    parameter WARPS_PER_CORE        = 1,
    parameter WARP_POLICY           = 0,
    parameter SIMT_STACK            = 0
) (
    input  wire                              clk,
    input  wire                              reset,
    input  wire                              telemetry_reset,
    input  wire                              start,
    output wire                              done,

    input  wire [7:0]                        block_id,
    input  wire [$clog2(THREADS_PER_BLOCK):0] thread_count,

    output wire                              program_mem_read_valid,
    output wire [PROGRAM_MEM_ADDR_BITS-1:0]  program_mem_read_address,
    input  wire                              program_mem_read_ready,
    input  wire [PROGRAM_MEM_DATA_BITS-1:0]  program_mem_read_data,

    output wire [THREADS_PER_BLOCK-1:0]      data_mem_read_valid,
    output wire [DATA_MEM_ADDR_BITS-1:0]     data_mem_read_address  [THREADS_PER_BLOCK-1:0],
    input  wire [THREADS_PER_BLOCK-1:0]      data_mem_read_ready,
    input  wire [DATA_MEM_DATA_BITS-1:0]     data_mem_read_data     [THREADS_PER_BLOCK-1:0],
    output wire [THREADS_PER_BLOCK-1:0]      data_mem_write_valid,
    output wire [DATA_MEM_ADDR_BITS-1:0]     data_mem_write_address [THREADS_PER_BLOCK-1:0],
    output wire [DATA_MEM_DATA_BITS-1:0]     data_mem_write_data    [THREADS_PER_BLOCK-1:0],
    input  wire [THREADS_PER_BLOCK-1:0]      data_mem_write_ready,

    output wire [31:0]                       icache_hits,
    output wire [31:0]                       icache_misses,
    output wire [31:0]                       warp_issues,
    output wire [31:0]                       bar_stall_cycles
);
    localparam LANES = THREADS_PER_BLOCK / WARPS_PER_CORE;

    wire [2:0]  core_state;
    wire [2:0]  fetcher_state;
    wire [15:0] instruction;

    wire [7:0]  current_pc;
    wire [7:0]  next_pc [THREADS_PER_BLOCK-1:0];
    wire [WORD_BITS-1:0] rs      [THREADS_PER_BLOCK-1:0];
    wire [WORD_BITS-1:0] rt      [THREADS_PER_BLOCK-1:0];
    wire [1:0]  lsu_state [THREADS_PER_BLOCK-1:0];
    wire [WORD_BITS-1:0] lsu_out [THREADS_PER_BLOCK-1:0];
    wire [WORD_BITS-1:0] alu_out [LANES-1:0];
    wire [$clog2(WARPS_PER_CORE+1)-1:0] current_warp;

    wire [3:0]  decoded_rd_address, decoded_rs_address, decoded_rt_address;
    wire [2:0]  decoded_nzp;
    wire [7:0]  decoded_immediate;
    wire        decoded_reg_write_enable, decoded_mem_read_enable, decoded_mem_write_enable;
    wire        decoded_nzp_write_enable;
    wire [1:0]  decoded_reg_input_mux;
    wire [2:0]  decoded_alu_arithmetic_mux;
    wire        decoded_alu_output_mux, decoded_pc_mux, decoded_ret;
    wire        decoded_smem_read_enable, decoded_smem_write_enable, decoded_bar;

    wire [THREADS_PER_BLOCK-1:0] lane_taken, lane_on;
    wire [7:0]  simt_next_pc;
    wire        simt_done;

    // Shared memory wires
    wire [THREADS_PER_BLOCK-1:0] smem_read_enable_v;
    wire [THREADS_PER_BLOCK-1:0] smem_write_enable_v;
    wire [7:0] smem_read_address  [THREADS_PER_BLOCK-1:0];
    wire [WORD_BITS-1:0] smem_read_data [THREADS_PER_BLOCK-1:0];
    wire [7:0] smem_write_address [THREADS_PER_BLOCK-1:0];
    wire [WORD_BITS-1:0] smem_write_data [THREADS_PER_BLOCK-1:0];
    wire [THREADS_PER_BLOCK-1:0] smem_ready_v;

    fetcher_mm #(
        .PROGRAM_MEM_ADDR_BITS(PROGRAM_MEM_ADDR_BITS),
        .PROGRAM_MEM_DATA_BITS(PROGRAM_MEM_DATA_BITS),
        .ICACHE_LINES(ICACHE_LINES)
    ) fetcher_instance (
        .clk(clk), .reset(reset), .telemetry_reset(telemetry_reset),
        .core_state(core_state), .current_pc(current_pc),
        .mem_read_valid(program_mem_read_valid),
        .mem_read_address(program_mem_read_address),
        .mem_read_ready(program_mem_read_ready),
        .mem_read_data(program_mem_read_data),
        .fetcher_state(fetcher_state),
        .instruction(instruction),
        .icache_hits(icache_hits),
        .icache_misses(icache_misses)
    );

    decoder decoder_instance (
        .clk(clk), .reset(reset),
        .core_state(core_state), .instruction(instruction),
        .decoded_rd_address(decoded_rd_address),
        .decoded_rs_address(decoded_rs_address),
        .decoded_rt_address(decoded_rt_address),
        .decoded_nzp(decoded_nzp),
        .decoded_immediate(decoded_immediate),
        .decoded_reg_write_enable(decoded_reg_write_enable),
        .decoded_mem_read_enable(decoded_mem_read_enable),
        .decoded_mem_write_enable(decoded_mem_write_enable),
        .decoded_nzp_write_enable(decoded_nzp_write_enable),
        .decoded_reg_input_mux(decoded_reg_input_mux),
        .decoded_alu_arithmetic_mux(decoded_alu_arithmetic_mux),
        .decoded_alu_output_mux(decoded_alu_output_mux),
        .decoded_pc_mux(decoded_pc_mux),
        .decoded_ret(decoded_ret),
        .decoded_smem_read_enable(decoded_smem_read_enable),
        .decoded_smem_write_enable(decoded_smem_write_enable),
        .decoded_bar(decoded_bar)
    );

    scheduler #(
        .THREADS_PER_BLOCK(THREADS_PER_BLOCK),
        .PIPELINED(PIPELINED),
        .WARPS_PER_CORE(WARPS_PER_CORE),
        .WARP_POLICY(WARP_POLICY),
        .SIMT_STACK(SIMT_STACK)
    ) scheduler_instance (
        .clk(clk), .reset(reset), .telemetry_reset(telemetry_reset), .start(start),
        .thread_count(thread_count),
        .decoded_mem_read_enable(decoded_mem_read_enable),
        .decoded_mem_write_enable(decoded_mem_write_enable),
        .decoded_smem_read_enable(decoded_smem_read_enable),
        .decoded_smem_write_enable(decoded_smem_write_enable),
        .decoded_bar(decoded_bar),
        .decoded_ret(decoded_ret),
        .fetcher_state(fetcher_state),
        .lsu_state(lsu_state),
        .current_pc(current_pc), .next_pc(next_pc),
        .simt_next_pc(simt_next_pc), .simt_done(simt_done),
        .core_state(core_state), .done(done),
        .current_warp(current_warp),
        .warp_issues(warp_issues),
        .bar_stall_cycles(bar_stall_cycles)
    );

    shared_memory_mm #(
        .THREADS_PER_BLOCK(THREADS_PER_BLOCK),
        .SIZE(SMEM_SIZE),
        .ADDR_BITS(8),
        .DATA_BITS(WORD_BITS)
    ) smem_instance (
        .clk(clk), .reset(reset),
        .lane_read_enable(smem_read_enable_v),
        .lane_read_addr(smem_read_address),
        .lane_read_data(smem_read_data),
        .lane_write_enable(smem_write_enable_v),
        .lane_write_addr(smem_write_address),
        .lane_write_data(smem_write_data),
        .lane_ready(smem_ready_v)
    );

    wire [3:0] decoded_mem_op = {decoded_mem_read_enable, decoded_mem_write_enable,
                                 decoded_smem_read_enable, decoded_smem_write_enable};

    genvar i;
    generate
        if (SIMT_STACK != 0) begin : simt
            wire [7:0]                stack_pc   [WARPS_PER_CORE-1:0];
            wire [WARPS_PER_CORE-1:0] stack_done;
            for (i = 0; i < WARPS_PER_CORE; i = i + 1) begin : warp
                wire [LANES-1:0] exists;
                genvar l;
                for (l = 0; l < LANES; l = l + 1) begin : lane
                    assign exists[l] = i * LANES + l < thread_count;
                end
                simt_stack #(.LANES(LANES), .DEPTH(8)) stack_instance (
                    .clk(clk), .reset(reset),
                    .update(core_state == 3'b110 && current_warp == i),
                    .exists(exists),
                    .branch(decoded_pc_mux && decoded_nzp != 3'b000),
                    .sync(decoded_pc_mux && decoded_nzp == 3'b000),
                    .ret(decoded_ret),
                    .immediate(decoded_immediate),
                    .taken(lane_taken[i * LANES +: LANES]),
                    .mask(lane_on[i * LANES +: LANES]),
                    .next_pc(stack_pc[i]),
                    .next_done(stack_done[i])
                );
            end
            assign simt_next_pc = stack_pc[current_warp];
            assign simt_done    = stack_done[current_warp];
        end else begin : no_simt
            assign lane_on      = {THREADS_PER_BLOCK{1'b1}};
            assign simt_next_pc = 0;
            assign simt_done    = 1'b0;
        end

        for (i = 0; i < LANES; i = i + 1) begin : lanes
            if (WARPS_PER_CORE == 1) begin : one_warp
                alu #(.WORD_BITS(WORD_BITS)) alu_instance (
                    .clk(clk), .reset(reset), .enable(i < thread_count),
                    .core_state(core_state),
                    .decoded_alu_arithmetic_mux(decoded_alu_arithmetic_mux),
                    .decoded_alu_output_mux(decoded_alu_output_mux),
                    .rs(rs[i]), .rt(rt[i]),
                    .alu_out(alu_out[i])
                );
            end else begin : warps
                wire [7:0] t = current_warp * LANES + i;
                alu #(.WORD_BITS(WORD_BITS)) alu_instance (
                    .clk(clk), .reset(reset), .enable(t < thread_count),
                    .core_state(core_state),
                    .decoded_alu_arithmetic_mux(decoded_alu_arithmetic_mux),
                    .decoded_alu_output_mux(decoded_alu_output_mux),
                    .rs(rs[t]), .rt(rt[t]),
                    .alu_out(alu_out[i])
                );
            end
        end

        for (i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin : threads
            // At several warps a thread is active only while its warp issues; its LSU sees the core only then
            // and keeps the memory operation it was issued (ADR-014).
            wire       active;
            wire [2:0] lsu_core_state;
            wire [3:0] mem_op;
            if (WARPS_PER_CORE == 1) begin : one_warp
                assign active         = i < thread_count && lane_on[i];
                assign lsu_core_state = lane_on[i] ? core_state : 3'b000;
                assign mem_op         = decoded_mem_op;
            end else begin : warps
                wire      issuing    = current_warp == i / LANES;
                wire      requesting = issuing && core_state == 3'b011;
                reg [3:0] issued_op;
                always @(posedge clk)
                    if (reset) issued_op <= 0;
                    else if (requesting) issued_op <= decoded_mem_op;
                assign active         = i < thread_count && issuing && lane_on[i];
                assign lsu_core_state = issuing && lane_on[i] ? core_state : 3'b000;
                assign mem_op         = requesting ? decoded_mem_op : issued_op;
            end

            lsu_mm #(
                .WORD_BITS(WORD_BITS),
                .ADDR_BITS(DATA_MEM_ADDR_BITS)
            ) lsu_instance (
                .clk(clk), .reset(reset), .enable(i < thread_count),
                .core_state(lsu_core_state),
                .decoded_mem_read_enable(mem_op[3]),
                .decoded_mem_write_enable(mem_op[2]),
                .decoded_smem_read_enable(mem_op[1]),
                .decoded_smem_write_enable(mem_op[0]),
                .rs(rs[i]), .rt(rt[i]),
                .mem_read_valid(data_mem_read_valid[i]),
                .mem_read_address(data_mem_read_address[i]),
                .mem_read_ready(data_mem_read_ready[i]),
                .mem_read_data(data_mem_read_data[i]),
                .mem_write_valid(data_mem_write_valid[i]),
                .mem_write_address(data_mem_write_address[i]),
                .mem_write_data(data_mem_write_data[i]),
                .mem_write_ready(data_mem_write_ready[i]),
                .smem_read_enable(smem_read_enable_v[i]),
                .smem_read_address(smem_read_address[i]),
                .smem_read_data(smem_read_data[i]),
                .smem_ready(smem_ready_v[i]),
                .smem_write_enable(smem_write_enable_v[i]),
                .smem_write_address(smem_write_address[i]),
                .smem_write_data(smem_write_data[i]),
                .lsu_state(lsu_state[i]),
                .lsu_out(lsu_out[i])
            );

            registers #(
                .THREADS_PER_BLOCK(THREADS_PER_BLOCK),
                .THREAD_ID(i),
                .WORD_BITS(WORD_BITS)
            ) register_instance (
                .clk(clk), .reset(reset), .enable(active),
                .block_id(block_id),
                .core_state(core_state),
                .decoded_reg_write_enable(decoded_reg_write_enable),
                .decoded_reg_input_mux(decoded_reg_input_mux),
                .decoded_rd_address(decoded_rd_address),
                .decoded_rs_address(decoded_rs_address),
                .decoded_rt_address(decoded_rt_address),
                .decoded_immediate(decoded_immediate),
                .alu_out(alu_out[i % LANES]),
                .lsu_out(lsu_out[i]),
                .rs(rs[i]), .rt(rt[i])
            );

            pc #(
                .WORD_BITS(WORD_BITS),
                .PROGRAM_MEM_ADDR_BITS(PROGRAM_MEM_ADDR_BITS)
            ) pc_instance (
                .clk(clk), .reset(reset), .enable(active),
                .core_state(core_state),
                .decoded_nzp(decoded_nzp),
                .decoded_immediate(decoded_immediate),
                .decoded_nzp_write_enable(decoded_nzp_write_enable),
                .decoded_pc_mux(decoded_pc_mux),
                .alu_out(alu_out[i % LANES]),
                .current_pc(current_pc),
                .next_pc(next_pc[i]),
                .taken(lane_taken[i])
            );
        end
    endgenerate
endmodule
