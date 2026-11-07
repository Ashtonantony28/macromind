`default_nettype none
`timescale 1ns/1ns
// DERIVED by designs/tinygpu/derive_rtl.py from upstream src/gpu.sv
// (Didactic_GPU_Simulator, commit in designs/tinygpu/UPSTREAM_PIN). Do not edit by hand.
// MacroMind change: module renamed gpu_mm; instantiates core_mm and l1d_cache_mm
//   (banked SRAM-macro L1-D). New parameter DCACHE_BANKS. Everything else is upstream.

// GPU Top
// Wires the dispatcher, memory controllers, and an array of cores onto external memory.
// All Phase-3+ extensions are gated by parameters so the original Phase-1 baseline is the default.
// WORD_BITS is the register/ALU/SMEM width and must equal DATA_MEM_DATA_BITS (ADR-010).
module gpu_mm #(
    parameter WORD_BITS                 = 8,
    parameter DATA_MEM_ADDR_BITS        = 8,
    parameter DATA_MEM_DATA_BITS        = 8,
    parameter DATA_MEM_NUM_CHANNELS     = 4,
    parameter PROGRAM_MEM_ADDR_BITS     = 8,
    parameter PROGRAM_MEM_DATA_BITS     = 16,
    parameter PROGRAM_MEM_NUM_CHANNELS  = 1,
    parameter NUM_CORES                 = 2,
    parameter THREADS_PER_BLOCK         = 4,
    parameter ICACHE_LINES              = 0,
    parameter COALESCE                  = 0,
    parameter PIPELINED                 = 0,
    parameter DCACHE_LINES              = 0,
    parameter DCACHE_BANKS              = 1,
    parameter MEDIC                     = 0,
    parameter WARPS_PER_CORE            = 1,
    parameter WARP_POLICY               = 0,
    parameter SIMT_STACK                = 0
) (
    input  wire                              clk,
    input  wire                              reset,

    input  wire                              start,
    output wire                              done,

    input  wire                              device_control_write_enable,
    input  wire [7:0]                        device_control_data,

    output wire [PROGRAM_MEM_NUM_CHANNELS-1:0] program_mem_read_valid,
    output wire [PROGRAM_MEM_ADDR_BITS-1:0]    program_mem_read_address [PROGRAM_MEM_NUM_CHANNELS-1:0],
    input  wire [PROGRAM_MEM_NUM_CHANNELS-1:0] program_mem_read_ready,
    input  wire [PROGRAM_MEM_DATA_BITS-1:0]    program_mem_read_data    [PROGRAM_MEM_NUM_CHANNELS-1:0],

    output wire [DATA_MEM_NUM_CHANNELS-1:0]    data_mem_read_valid,
    output wire [DATA_MEM_ADDR_BITS-1:0]       data_mem_read_address    [DATA_MEM_NUM_CHANNELS-1:0],
    input  wire [DATA_MEM_NUM_CHANNELS-1:0]    data_mem_read_ready,
    input  wire [DATA_MEM_DATA_BITS-1:0]       data_mem_read_data       [DATA_MEM_NUM_CHANNELS-1:0],
    output wire [DATA_MEM_NUM_CHANNELS-1:0]    data_mem_write_valid,
    output wire [DATA_MEM_ADDR_BITS-1:0]       data_mem_write_address   [DATA_MEM_NUM_CHANNELS-1:0],
    output wire [DATA_MEM_DATA_BITS-1:0]       data_mem_write_data      [DATA_MEM_NUM_CHANNELS-1:0],
    input  wire [DATA_MEM_NUM_CHANNELS-1:0]    data_mem_write_ready,

    // Bench telemetry (Phase 2+)
    output wire [31:0]                         data_mem_requests,
    output wire [31:0]                         data_mem_requests_coalesced,
    output wire [31:0]                         icache_hits_total,
    output wire [31:0]                         icache_misses_total,
    output wire [31:0]                         dcache_hits_total,
    output wire [31:0]                         dcache_misses_total,
    output wire [31:0]                         warp_issues_total,
    output wire [31:0]                         bar_stall_cycles_total
);
    initial if (WORD_BITS != DATA_MEM_DATA_BITS) $fatal(1, "gpu: WORD_BITS (%0d) != DATA_MEM_DATA_BITS (%0d)", WORD_BITS, DATA_MEM_DATA_BITS);
    initial if (WARPS_PER_CORE < 1 || THREADS_PER_BLOCK % WARPS_PER_CORE != 0)
        $fatal(1, "gpu: WARPS_PER_CORE (%0d) does not divide THREADS_PER_BLOCK (%0d)", WARPS_PER_CORE, THREADS_PER_BLOCK);
    initial if (WARP_POLICY < 0 || WARP_POLICY > 2) $fatal(1, "gpu: WARP_POLICY (%0d) is not 0, 1 or 2", WARP_POLICY);

    wire [7:0] thread_count;

    wire [NUM_CORES-1:0] core_start;
    wire [NUM_CORES-1:0] core_reset;
    wire [NUM_CORES-1:0] core_done;
    wire [7:0]           core_block_id     [NUM_CORES-1:0];
    wire [$clog2(THREADS_PER_BLOCK):0] core_thread_count [NUM_CORES-1:0];

    localparam NUM_LSUS = NUM_CORES * THREADS_PER_BLOCK;
    wire [NUM_LSUS-1:0] lsu_read_valid;
    wire [DATA_MEM_ADDR_BITS-1:0] lsu_read_address [NUM_LSUS-1:0];
    wire [NUM_LSUS-1:0] lsu_read_ready;
    wire [DATA_MEM_DATA_BITS-1:0] lsu_read_data    [NUM_LSUS-1:0];
    wire [NUM_LSUS-1:0] lsu_write_valid;
    wire [DATA_MEM_ADDR_BITS-1:0] lsu_write_address [NUM_LSUS-1:0];
    wire [DATA_MEM_DATA_BITS-1:0] lsu_write_data    [NUM_LSUS-1:0];
    wire [NUM_LSUS-1:0] lsu_write_ready;

    localparam NUM_FETCHERS = NUM_CORES;
    wire [NUM_FETCHERS-1:0]              fetcher_read_valid;
    wire [PROGRAM_MEM_ADDR_BITS-1:0]     fetcher_read_address [NUM_FETCHERS-1:0];
    wire [NUM_FETCHERS-1:0]              fetcher_read_ready;
    wire [PROGRAM_MEM_DATA_BITS-1:0]     fetcher_read_data    [NUM_FETCHERS-1:0];

    // Program controller doesn't write, but the controller wants a write port. Wire to 0.
    wire [NUM_FETCHERS-1:0] zero_write_valid = '0;
    wire [PROGRAM_MEM_ADDR_BITS-1:0] zero_write_addr [NUM_FETCHERS-1:0];
    wire [PROGRAM_MEM_DATA_BITS-1:0] zero_write_data [NUM_FETCHERS-1:0];
    wire [NUM_FETCHERS-1:0] discard_write_ready;
    wire [PROGRAM_MEM_NUM_CHANNELS-1:0] discard_pmem_write_valid;
    wire [PROGRAM_MEM_ADDR_BITS-1:0] discard_pmem_write_addr [PROGRAM_MEM_NUM_CHANNELS-1:0];
    wire [PROGRAM_MEM_DATA_BITS-1:0] discard_pmem_write_data [PROGRAM_MEM_NUM_CHANNELS-1:0];
    wire [PROGRAM_MEM_NUM_CHANNELS-1:0] zero_pmem_write_ready = '0;
    wire [31:0] prog_requests, prog_requests_coalesced;

    genvar gi;
    generate
        for (gi = 0; gi < NUM_FETCHERS; gi = gi + 1) begin : zwa
            assign zero_write_addr[gi] = '0;
            assign zero_write_data[gi] = '0;
        end
    endgenerate

    dcr dcr_instance (
        .clk(clk), .reset(reset),
        .device_control_write_enable(device_control_write_enable),
        .device_control_data(device_control_data),
        .thread_count(thread_count)
    );

    // L1-D cache layer (Phase 9). When DCACHE_LINES=0 we bypass entirely.
    wire [NUM_LSUS-1:0]            ctl_read_valid;
    wire [DATA_MEM_ADDR_BITS-1:0]  ctl_read_address [NUM_LSUS-1:0];
    wire [NUM_LSUS-1:0]            ctl_read_ready;
    wire [DATA_MEM_DATA_BITS-1:0]  ctl_read_data    [NUM_LSUS-1:0];
    wire [NUM_LSUS-1:0]            ctl_write_valid;
    wire [DATA_MEM_ADDR_BITS-1:0]  ctl_write_address [NUM_LSUS-1:0];
    wire [DATA_MEM_DATA_BITS-1:0]  ctl_write_data    [NUM_LSUS-1:0];
    wire [NUM_LSUS-1:0]            ctl_write_ready;
    wire [31:0]                    dcache_hits_w, dcache_misses_w;
    wire [NUM_LSUS-1:0]            warp_low_util, warp_high_util;

    generate
        if (DCACHE_LINES > 0) begin : with_l1d
            l1d_cache_mm #(
                .NUM_CONSUMERS(NUM_LSUS),
                .ADDR_BITS(DATA_MEM_ADDR_BITS),
                .DATA_BITS(DATA_MEM_DATA_BITS),
                .LINES(DCACHE_LINES),
                .BANKS(DCACHE_BANKS),
                .MEDIC(MEDIC)
            ) l1d (
                .clk(clk), .reset(reset), .telemetry_reset(reset),
                .consumer_read_valid(lsu_read_valid),
                .consumer_read_address(lsu_read_address),
                .consumer_read_ready(lsu_read_ready),
                .consumer_read_data(lsu_read_data),
                .consumer_write_valid(lsu_write_valid),
                .consumer_write_address(lsu_write_address),
                .consumer_write_data(lsu_write_data),
                .consumer_write_ready(lsu_write_ready),
                .mem_read_valid(ctl_read_valid),
                .mem_read_address(ctl_read_address),
                .mem_read_ready(ctl_read_ready),
                .mem_read_data(ctl_read_data),
                .mem_write_valid(ctl_write_valid),
                .mem_write_address(ctl_write_address),
                .mem_write_data(ctl_write_data),
                .mem_write_ready(ctl_write_ready),
                .cache_hits_total(dcache_hits_w),
                .cache_misses_total(dcache_misses_w),
                .warp_low_utility(warp_low_util),
                .warp_high_utility(warp_high_util)
            );
        end else begin : no_l1d
            assign ctl_read_valid     = lsu_read_valid;
            assign lsu_read_ready     = ctl_read_ready;
            assign ctl_write_valid    = lsu_write_valid;
            assign lsu_write_ready    = ctl_write_ready;
            genvar k;
            for (k = 0; k < NUM_LSUS; k = k + 1) begin : passthru
                assign ctl_read_address[k]   = lsu_read_address[k];
                assign lsu_read_data[k]      = ctl_read_data[k];
                assign ctl_write_address[k]  = lsu_write_address[k];
                assign ctl_write_data[k]     = lsu_write_data[k];
            end
            assign dcache_hits_w   = 32'b0;
            assign dcache_misses_w = 32'b0;
            assign warp_low_util   = {NUM_LSUS{1'b0}};
            assign warp_high_util  = {NUM_LSUS{1'b0}};
        end
    endgenerate

    assign dcache_hits_total   = dcache_hits_w;
    assign dcache_misses_total = dcache_misses_w;

    controller #(
        .ADDR_BITS(DATA_MEM_ADDR_BITS),
        .DATA_BITS(DATA_MEM_DATA_BITS),
        .NUM_CONSUMERS(NUM_LSUS),
        .NUM_CHANNELS(DATA_MEM_NUM_CHANNELS),
        .WRITE_ENABLE(1),
        .COALESCE(COALESCE),
        .PRIORITY_ENABLE(MEDIC)
    ) data_memory_controller (
        .clk(clk), .reset(reset),
        .consumer_read_valid(ctl_read_valid),
        .consumer_read_address(ctl_read_address),
        .consumer_read_ready(ctl_read_ready),
        .consumer_read_data(ctl_read_data),
        .consumer_write_valid(ctl_write_valid),
        .consumer_write_address(ctl_write_address),
        .consumer_write_data(ctl_write_data),
        .consumer_write_ready(ctl_write_ready),
        .consumer_priority(warp_high_util),
        .mem_read_valid(data_mem_read_valid),
        .mem_read_address(data_mem_read_address),
        .mem_read_ready(data_mem_read_ready),
        .mem_read_data(data_mem_read_data),
        .mem_write_valid(data_mem_write_valid),
        .mem_write_address(data_mem_write_address),
        .mem_write_data(data_mem_write_data),
        .mem_write_ready(data_mem_write_ready),
        .mem_requests(data_mem_requests),
        .mem_requests_coalesced(data_mem_requests_coalesced)
    );

    wire [NUM_FETCHERS-1:0] zero_priority = '0;
    controller #(
        .ADDR_BITS(PROGRAM_MEM_ADDR_BITS),
        .DATA_BITS(PROGRAM_MEM_DATA_BITS),
        .NUM_CONSUMERS(NUM_FETCHERS),
        .NUM_CHANNELS(PROGRAM_MEM_NUM_CHANNELS),
        .WRITE_ENABLE(0),
        .COALESCE(0),
        .PRIORITY_ENABLE(0)
    ) program_memory_controller (
        .clk(clk), .reset(reset),
        .consumer_read_valid(fetcher_read_valid),
        .consumer_read_address(fetcher_read_address),
        .consumer_read_ready(fetcher_read_ready),
        .consumer_read_data(fetcher_read_data),
        .consumer_write_valid(zero_write_valid),
        .consumer_write_address(zero_write_addr),
        .consumer_write_data(zero_write_data),
        .consumer_write_ready(discard_write_ready),
        .consumer_priority(zero_priority),
        .mem_read_valid(program_mem_read_valid),
        .mem_read_address(program_mem_read_address),
        .mem_read_ready(program_mem_read_ready),
        .mem_read_data(program_mem_read_data),
        .mem_write_valid(discard_pmem_write_valid),
        .mem_write_address(discard_pmem_write_addr),
        .mem_write_data(discard_pmem_write_data),
        .mem_write_ready(zero_pmem_write_ready),
        .mem_requests(prog_requests),
        .mem_requests_coalesced(prog_requests_coalesced)
    );

    dispatch #(
        .NUM_CORES(NUM_CORES),
        .THREADS_PER_BLOCK(THREADS_PER_BLOCK)
    ) dispatch_instance (
        .clk(clk), .reset(reset), .start(start),
        .thread_count(thread_count),
        .core_done(core_done),
        .core_start(core_start),
        .core_reset(core_reset),
        .core_block_id(core_block_id),
        .core_thread_count(core_thread_count),
        .done(done)
    );

    // Aggregate per-core telemetry (packed for portability).
    wire [NUM_CORES*32-1:0] per_core_icache_hits;
    wire [NUM_CORES*32-1:0] per_core_icache_misses;
    wire [NUM_CORES*32-1:0] per_core_warp_issues;
    wire [NUM_CORES*32-1:0] per_core_bar_stall_cycles;

    genvar c;
    generate
        for (c = 0; c < NUM_CORES; c = c + 1) begin : cores
            wire [THREADS_PER_BLOCK-1:0]      core_lsu_read_valid;
            wire [DATA_MEM_ADDR_BITS-1:0]     core_lsu_read_address  [THREADS_PER_BLOCK-1:0];
            wire [THREADS_PER_BLOCK-1:0]      core_lsu_read_ready;
            wire [DATA_MEM_DATA_BITS-1:0]     core_lsu_read_data     [THREADS_PER_BLOCK-1:0];
            wire [THREADS_PER_BLOCK-1:0]      core_lsu_write_valid;
            wire [DATA_MEM_ADDR_BITS-1:0]     core_lsu_write_address [THREADS_PER_BLOCK-1:0];
            wire [DATA_MEM_DATA_BITS-1:0]     core_lsu_write_data    [THREADS_PER_BLOCK-1:0];
            wire [THREADS_PER_BLOCK-1:0]      core_lsu_write_ready;

            genvar t;
            for (t = 0; t < THREADS_PER_BLOCK; t = t + 1) begin : lsu_xbar
                localparam integer LIDX = c * THREADS_PER_BLOCK + t;
                assign lsu_read_valid[LIDX]      = core_lsu_read_valid[t];
                assign lsu_read_address[LIDX]    = core_lsu_read_address[t];
                assign core_lsu_read_ready[t]    = lsu_read_ready[LIDX];
                assign core_lsu_read_data[t]     = lsu_read_data[LIDX];
                assign lsu_write_valid[LIDX]     = core_lsu_write_valid[t];
                assign lsu_write_address[LIDX]   = core_lsu_write_address[t];
                assign lsu_write_data[LIDX]      = core_lsu_write_data[t];
                assign core_lsu_write_ready[t]   = lsu_write_ready[LIDX];
            end

            core_mm #(
                .DATA_MEM_ADDR_BITS(DATA_MEM_ADDR_BITS),
                .DATA_MEM_DATA_BITS(DATA_MEM_DATA_BITS),
                .WORD_BITS(WORD_BITS),
                .PROGRAM_MEM_ADDR_BITS(PROGRAM_MEM_ADDR_BITS),
                .PROGRAM_MEM_DATA_BITS(PROGRAM_MEM_DATA_BITS),
                .THREADS_PER_BLOCK(THREADS_PER_BLOCK),
                .ICACHE_LINES(ICACHE_LINES),
                .PIPELINED(PIPELINED),
                .WARPS_PER_CORE(WARPS_PER_CORE),
                .WARP_POLICY(WARP_POLICY),
                .SIMT_STACK(SIMT_STACK)
            ) core_instance (
                .clk(clk),
                .reset(core_reset[c]),
                .telemetry_reset(reset),
                .start(core_start[c]),
                .done(core_done[c]),
                .block_id(core_block_id[c]),
                .thread_count(core_thread_count[c]),
                .program_mem_read_valid(fetcher_read_valid[c]),
                .program_mem_read_address(fetcher_read_address[c]),
                .program_mem_read_ready(fetcher_read_ready[c]),
                .program_mem_read_data(fetcher_read_data[c]),
                .data_mem_read_valid(core_lsu_read_valid),
                .data_mem_read_address(core_lsu_read_address),
                .data_mem_read_ready(core_lsu_read_ready),
                .data_mem_read_data(core_lsu_read_data),
                .data_mem_write_valid(core_lsu_write_valid),
                .data_mem_write_address(core_lsu_write_address),
                .data_mem_write_data(core_lsu_write_data),
                .data_mem_write_ready(core_lsu_write_ready),
                .icache_hits(per_core_icache_hits[c*32 +: 32]),
                .icache_misses(per_core_icache_misses[c*32 +: 32]),
                .warp_issues(per_core_warp_issues[c*32 +: 32]),
                .bar_stall_cycles(per_core_bar_stall_cycles[c*32 +: 32])
            );
        end
    endgenerate

    // Sum over all NUM_CORES.
    reg [31:0] icache_hits_sum, icache_misses_sum, warp_issues_sum, bar_stall_cycles_sum;
    integer ci, cw;
    always @(*) begin
        icache_hits_sum   = 0;
        icache_misses_sum = 0;
        for (ci = 0; ci < NUM_CORES; ci = ci + 1) begin
            icache_hits_sum   = icache_hits_sum   + per_core_icache_hits  [ci*32 +: 32];
            icache_misses_sum = icache_misses_sum + per_core_icache_misses[ci*32 +: 32];
        end
    end
    always @(*) begin
        warp_issues_sum      = 0;
        bar_stall_cycles_sum = 0;
        for (cw = 0; cw < NUM_CORES; cw = cw + 1) begin
            warp_issues_sum      = warp_issues_sum      + per_core_warp_issues     [cw*32 +: 32];
            bar_stall_cycles_sum = bar_stall_cycles_sum + per_core_bar_stall_cycles[cw*32 +: 32];
        end
    end
    assign icache_hits_total      = icache_hits_sum;
    assign icache_misses_total    = icache_misses_sum;
    assign warp_issues_total      = warp_issues_sum;
    assign bar_stall_cycles_total = bar_stall_cycles_sum;
endmodule
