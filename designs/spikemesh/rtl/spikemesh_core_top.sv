// MacroMind: synthesis top for the SpikeMesh neurocore macro variant. Fixes the
// neurocore parameters for one configuration (SM_N / SM_A / SM_FD / SM_LEARN defines,
// set by configs.mk) and passes every port straight through.
`default_nettype none
`ifndef SM_N
`define SM_N 128
`endif
`ifndef SM_A
`define SM_A 128
`endif
`ifndef SM_FD
`define SM_FD 256
`endif
`ifndef SM_LEARN
`define SM_LEARN 0
`endif
module spikemesh_core_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    output wire        in_ready,
    input  wire [47:0] in_flit,
    output wire        out_valid,
    input  wire        out_ready,
    output wire [47:0] out_flit,
    input  wire        parity,
    input  wire        learn_start,
    input  wire        update_start,
    output wire        busy
);
    neurocore #(
        .X(0), .Y(0), .N_NEURONS(`SM_N), .N_AXONS(`SM_A), .W_BITS(8), .FANOUT_DEPTH(`SM_FD),
        .LANES(8), .LEARN(`SM_LEARN), .ADAPT_TH(1)
    ) u_core (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_ready(in_ready), .in_flit(in_flit),
        .out_valid(out_valid), .out_ready(out_ready), .out_flit(out_flit), .parity(parity),
        .learn_start(learn_start), .update_start(update_start), .busy(busy)
    );
endmodule
`default_nettype wire
