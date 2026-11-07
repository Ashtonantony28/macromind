`default_nettype none
`timescale 1ns/1ns

// MacroMind top-level wrapper for tiny-gpu-extended.
// Flattens the gpu's unpacked-array ports into packed buses (one slice per
// channel) so ORFS/OpenROAD see plain ports, and drops the bench telemetry
// outputs (so their counters are optimised away).
// Configuration comes from `defines (ORFS VERILOG_DEFINES) or parameters:
//   TG_MACROS=0  upstream gpu (flop memories)          TG_MACROS=1  gpu_mm (SRAM macros)
//   TG_WARPS     upstream WARPS_PER_CORE: 4 warps of 1 lane time-share ONE ALU per core
//                (ADR-012/014), which keeps the 32-bit divider/multiplier count = NUM_CORES.
module tinygpu_top #(
`ifndef TG_MACROS
  `define TG_MACROS 1
`endif
`ifndef TG_NUM_CORES
  `define TG_NUM_CORES 4
`endif
`ifndef TG_WORD_BITS
  `define TG_WORD_BITS 32
`endif
`ifndef TG_DATA_ADDR_BITS
  `define TG_DATA_ADDR_BITS 12
`endif
`ifndef TG_ICACHE_LINES
  `define TG_ICACHE_LINES 64
`endif
`ifndef TG_DCACHE_LINES
  `define TG_DCACHE_LINES 1024
`endif
`ifndef TG_DCACHE_BANKS
  `define TG_DCACHE_BANKS 4
`endif
`ifndef TG_WARPS
  `define TG_WARPS 4
`endif
`ifndef TG_DATA_CHANNELS
  `define TG_DATA_CHANNELS 4
`endif
    parameter MACROS            = `TG_MACROS,
    parameter NUM_CORES         = `TG_NUM_CORES,
    parameter WORD_BITS         = `TG_WORD_BITS,
    parameter DATA_ADDR_BITS    = `TG_DATA_ADDR_BITS,
    parameter ICACHE_LINES      = `TG_ICACHE_LINES,
    parameter DCACHE_LINES      = `TG_DCACHE_LINES,
    parameter DCACHE_BANKS      = `TG_DCACHE_BANKS,
    parameter DCH               = `TG_DATA_CHANNELS,
    parameter WARPS             = `TG_WARPS,
    parameter PADDR             = 8,
    parameter PDATA             = 16
) (
    input  wire                       clk,
    input  wire                       reset,
    input  wire                       start,
    output wire                       done,
    input  wire                       device_control_write_enable,
    input  wire [7:0]                 device_control_data,

    output wire                       program_mem_read_valid,
    output wire [PADDR-1:0]           program_mem_read_address,
    input  wire                       program_mem_read_ready,
    input  wire [PDATA-1:0]           program_mem_read_data,

    output wire [DCH-1:0]             data_mem_read_valid,
    output wire [DCH*DATA_ADDR_BITS-1:0] data_mem_read_address,
    input  wire [DCH-1:0]             data_mem_read_ready,
    input  wire [DCH*WORD_BITS-1:0]   data_mem_read_data,
    output wire [DCH-1:0]             data_mem_write_valid,
    output wire [DCH*DATA_ADDR_BITS-1:0] data_mem_write_address,
    output wire [DCH*WORD_BITS-1:0]   data_mem_write_data,
    input  wire [DCH-1:0]             data_mem_write_ready
);
    wire [PADDR-1:0]          p_addr [0:0];
    wire [PDATA-1:0]          p_data [0:0];
    wire [DATA_ADDR_BITS-1:0] d_raddr [DCH-1:0];
    wire [WORD_BITS-1:0]      d_rdata [DCH-1:0];
    wire [DATA_ADDR_BITS-1:0] d_waddr [DCH-1:0];
    wire [WORD_BITS-1:0]      d_wdata [DCH-1:0];
    wire [31:0] t0, t1, t2, t3, t4, t5, t6, t7;

    assign program_mem_read_address = p_addr[0];
    assign p_data[0] = program_mem_read_data;
    genvar c;
    generate
        for (c = 0; c < DCH; c = c + 1) begin : ch
            assign data_mem_read_address [c*DATA_ADDR_BITS +: DATA_ADDR_BITS] = d_raddr[c];
            assign data_mem_write_address[c*DATA_ADDR_BITS +: DATA_ADDR_BITS] = d_waddr[c];
            assign data_mem_write_data   [c*WORD_BITS +: WORD_BITS]           = d_wdata[c];
            assign d_rdata[c] = data_mem_read_data[c*WORD_BITS +: WORD_BITS];
        end
    endgenerate

`define TG_GPU_PORTS \
        .clk(clk), .reset(reset), .start(start), .done(done), \
        .device_control_write_enable(device_control_write_enable), \
        .device_control_data(device_control_data), \
        .program_mem_read_valid(program_mem_read_valid), .program_mem_read_address(p_addr), \
        .program_mem_read_ready(program_mem_read_ready), .program_mem_read_data(p_data), \
        .data_mem_read_valid(data_mem_read_valid), .data_mem_read_address(d_raddr), \
        .data_mem_read_ready(data_mem_read_ready), .data_mem_read_data(d_rdata), \
        .data_mem_write_valid(data_mem_write_valid), .data_mem_write_address(d_waddr), \
        .data_mem_write_data(d_wdata), .data_mem_write_ready(data_mem_write_ready), \
        .data_mem_requests(t0), .data_mem_requests_coalesced(t1), \
        .icache_hits_total(t2), .icache_misses_total(t3), \
        .dcache_hits_total(t4), .dcache_misses_total(t5), \
        .warp_issues_total(t6), .bar_stall_cycles_total(t7)

    generate
        if (MACROS != 0) begin : mm
            gpu_mm #(
                .WORD_BITS(WORD_BITS), .DATA_MEM_ADDR_BITS(DATA_ADDR_BITS),
                .DATA_MEM_DATA_BITS(WORD_BITS), .DATA_MEM_NUM_CHANNELS(DCH),
                .NUM_CORES(NUM_CORES), .ICACHE_LINES(ICACHE_LINES), .WARPS_PER_CORE(WARPS),
                .DCACHE_LINES(DCACHE_LINES), .DCACHE_BANKS(DCACHE_BANKS)
            ) u_gpu (`TG_GPU_PORTS);
        end else begin : ff
            gpu #(
                .WORD_BITS(WORD_BITS), .DATA_MEM_ADDR_BITS(DATA_ADDR_BITS),
                .DATA_MEM_DATA_BITS(WORD_BITS), .DATA_MEM_NUM_CHANNELS(DCH),
                .NUM_CORES(NUM_CORES), .ICACHE_LINES(ICACHE_LINES), .WARPS_PER_CORE(WARPS),
                .DCACHE_LINES(DCACHE_LINES)
            ) u_gpu (`TG_GPU_PORTS);
        end
    endgenerate
endmodule
