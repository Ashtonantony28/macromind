// MacroMind: drop-in replacement for SpikeMesh's rtl/core/sram_1r1w.sv (same module
// name, parameters, ports and behaviour) that maps a memory onto banks of the sky130
// OpenRAM macro sky130_sram_1rw1r_64x256_8 (port 0 writes, port 1 reads: a true
// 1R1W, which the neurocore needs; it reads and writes the same memory in one cycle).
//
// A memory is mapped when DEPTH is a multiple of 256, WIDTH <= 64, and its write
// lanes are bytes (WIDTH / NLANE == 8) or it has a single lane; otherwise it stays a
// flop array exactly like upstream. FORCE_FLOPS=1 disables the mapping.
//
// Upstream contract kept: registered read data, held while `re` is low; read and
// write in the same cycle allowed (same-address read returns old or new data: the
// neurocore never relies on it, see upstream sram_1r1w.sv). The macro drives dout
// only in the cycle after a read, so a hold register keeps rdata stable afterwards.
`default_nettype none

module sram_1r1w #(
    parameter integer DEPTH = 256,
    parameter integer WIDTH = 16,
    parameter integer NLANE = 1,
    parameter integer AW    = (DEPTH > 1) ? $clog2(DEPTH) : 1,
    parameter integer FORCE_FLOPS = 0
) (
    input  wire              clk,
    input  wire              we,
    input  wire [NLANE-1:0]  wmask,
    input  wire [AW-1:0]     waddr,
    input  wire [WIDTH-1:0]  wdata,
    input  wire              re,
    input  wire [AW-1:0]     raddr,
    output reg  [WIDTH-1:0]  rdata
);
    localparam integer LW       = WIDTH / NLANE;
    localparam integer M_DEPTH  = 256;            // sky130_sram_1rw1r_64x256_8
    localparam integer M_WIDTH  = 64;
    localparam integer M_AW     = 8;
    localparam integer M_LANES  = 8;              // byte write mask
    localparam integer USE_MACRO = (FORCE_FLOPS == 0) && (DEPTH % M_DEPTH == 0) && (WIDTH <= M_WIDTH)
                                   && ((NLANE == 1) || (LW == 8));
    localparam integer NBANK    = USE_MACRO ? DEPTH / M_DEPTH : 1;
    localparam integer BW       = (NBANK > 1) ? $clog2(NBANK) : 1;

    generate
        if (USE_MACRO == 0) begin : g_flops
            reg [WIDTH-1:0] mem [0:DEPTH-1];
            integer i;
            always @(posedge clk) begin
                if (we) begin
                    for (i = 0; i < NLANE; i = i + 1)
                        if (wmask[i]) mem[waddr][i*LW +: LW] <= wdata[i*LW +: LW];
                end
                if (re) rdata <= mem[raddr];
            end
        end else begin : g_macro
            // byte mask: lane i of NLANE byte lanes -> bit i; one lane -> all bytes it covers
            wire [M_LANES-1:0] bmask;
            wire [M_WIDTH-1:0] din = {{(M_WIDTH-WIDTH){1'b0}}, wdata};
            if (NLANE == 1) begin : g_m1
                assign bmask = {M_LANES{wmask[0]}};
            end else begin : g_mb
                assign bmask = {{(M_LANES-NLANE){1'b0}}, wmask};
            end
            wire [BW-1:0] wbank = (NBANK > 1) ? BW'(waddr >> M_AW) : '0;
            wire [BW-1:0] rbank = (NBANK > 1) ? BW'(raddr >> M_AW) : '0;
            wire [M_WIDTH-1:0] dout [0:NBANK-1];
            genvar b;
            for (b = 0; b < NBANK; b = b + 1) begin : g_bank
                sky130_sram_1rw1r_64x256_8 u_mem (
                    .clk0(clk), .csb0(!(we && (wbank == BW'(b)))), .web0(1'b0), .wmask0(bmask),
                    .addr0(waddr[M_AW-1:0]), .din0(din), .dout0(),
                    .clk1(clk), .csb1(!(re && (rbank == BW'(b)))), .addr1(raddr[M_AW-1:0]),
                    .dout1(dout[b]));
            end
            reg          rd_q;
            reg [BW-1:0] rbank_q;
            reg [WIDTH-1:0] hold;
            always @(posedge clk) begin
                rd_q <= re;
                if (re) rbank_q <= rbank;
                hold <= rdata;
            end
            always @(*) rdata = rd_q ? dout[rbank_q][WIDTH-1:0] : hold;
        end
    endgenerate
endmodule

`default_nettype wire
