`default_nettype none
`timescale 1ns/1ns

// MacroMind: generic single-port (1RW) synchronous SRAM, mapped onto the
// smallest NanGate45 fakeram45 macro with exactly DEPTH words and >= WIDTH bits
// (unused high bits tied off). Read data appears on rdata the cycle after
// ce & !we and holds until the next read. Write is ce & we (all bits).
// With no fitting macro (or FORCE_FLOPS=1) a flop array with the same
// timing is used instead. MACRO_W reports the chosen macro width (0 = flops).
// Generated macro table: see designs/tinygpu/docs or `fakeram45_*` in ORFS
// platforms/nangate45/lef.
module mm_sram #(
    parameter DEPTH       = 64,
    parameter WIDTH       = 32,
    parameter FORCE_FLOPS = 0
) (
    input  wire                     clk,
    input  wire                     ce,
    input  wire                     we,
    input  wire [$clog2(DEPTH)-1:0] addr,
    input  wire [WIDTH-1:0]         wdata,
    output wire [WIDTH-1:0]         rdata
);
    function automatic integer pick_width(input integer d, input integer w);
        integer best;
        begin
            best = 0;
            if (d == 32 && w <= 64) best = 64;
            if (d == 32 && w <= 32) best = 32;
            if (d == 64 && w <= 124) best = 124;
            if (d == 64 && w <= 96) best = 96;
            if (d == 64 && w <= 64) best = 64;
            if (d == 64 && w <= 62) best = 62;
            if (d == 64 && w <= 32) best = 32;
            if (d == 64 && w <= 21) best = 21;
            if (d == 64 && w <= 15) best = 15;
            if (d == 64 && w <= 7) best = 7;
            if (d == 128 && w <= 256) best = 256;
            if (d == 128 && w <= 116) best = 116;
            if (d == 128 && w <= 32) best = 32;
            if (d == 256 && w <= 96) best = 96;
            if (d == 256 && w <= 95) best = 95;
            if (d == 256 && w <= 48) best = 48;
            if (d == 256 && w <= 34) best = 34;
            if (d == 256 && w <= 32) best = 32;
            if (d == 256 && w <= 16) best = 16;
            if (d == 512 && w <= 64) best = 64;
            if (d == 1024 && w <= 32) best = 32;
            if (d == 2048 && w <= 39) best = 39;
            pick_width = best;
        end
    endfunction

    localparam integer MACRO_W = FORCE_FLOPS ? 0 : pick_width(DEPTH, WIDTH);
    localparam integer AW = $clog2(DEPTH);

    generate
        if (MACRO_W == 0) begin : flops
            reg [WIDTH-1:0] mem [0:DEPTH-1];
            reg [WIDTH-1:0] q;
            always @(posedge clk) begin
                if (ce && we)  mem[addr] <= wdata;
                if (ce && !we) q <= mem[addr];
            end
            assign rdata = q;
        end else begin : macro
            wire [MACRO_W-1:0] wd;
            assign wd = wdata;  // zero-extend into the macro width
            wire [MACRO_W-1:0] rd;
            assign rdata = rd[WIDTH-1:0];
            if (DEPTH == 32 && MACRO_W == 32) begin : m32x32
                fakeram45_32x32 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({32{1'b1}}));
            end
            else if (DEPTH == 32 && MACRO_W == 64) begin : m32x64
                fakeram45_32x64 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({64{1'b1}}));
            end
            else if (DEPTH == 64 && MACRO_W == 7) begin : m64x7
                fakeram45_64x7 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({7{1'b1}}));
            end
            else if (DEPTH == 64 && MACRO_W == 15) begin : m64x15
                fakeram45_64x15 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({15{1'b1}}));
            end
            else if (DEPTH == 64 && MACRO_W == 21) begin : m64x21
                fakeram45_64x21 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({21{1'b1}}));
            end
            else if (DEPTH == 64 && MACRO_W == 32) begin : m64x32
                fakeram45_64x32 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({32{1'b1}}));
            end
            else if (DEPTH == 64 && MACRO_W == 62) begin : m64x62
                fakeram45_64x62 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({62{1'b1}}));
            end
            else if (DEPTH == 64 && MACRO_W == 64) begin : m64x64
                fakeram45_64x64 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({64{1'b1}}));
            end
            else if (DEPTH == 64 && MACRO_W == 96) begin : m64x96
                fakeram45_64x96 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({96{1'b1}}));
            end
            else if (DEPTH == 64 && MACRO_W == 124) begin : m64x124
                fakeram45_64x124 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({124{1'b1}}));
            end
            else if (DEPTH == 128 && MACRO_W == 32) begin : m128x32
                fakeram45_128x32 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({32{1'b1}}));
            end
            else if (DEPTH == 128 && MACRO_W == 116) begin : m128x116
                fakeram45_128x116 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({116{1'b1}}));
            end
            else if (DEPTH == 128 && MACRO_W == 256) begin : m128x256
                fakeram45_128x256 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({256{1'b1}}));
            end
            else if (DEPTH == 256 && MACRO_W == 16) begin : m256x16
                fakeram45_256x16 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({16{1'b1}}));
            end
            else if (DEPTH == 256 && MACRO_W == 32) begin : m256x32
                fakeram45_256x32 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({32{1'b1}}));
            end
            else if (DEPTH == 256 && MACRO_W == 34) begin : m256x34
                fakeram45_256x34 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({34{1'b1}}));
            end
            else if (DEPTH == 256 && MACRO_W == 48) begin : m256x48
                fakeram45_256x48 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({48{1'b1}}));
            end
            else if (DEPTH == 256 && MACRO_W == 95) begin : m256x95
                fakeram45_256x95 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({95{1'b1}}));
            end
            else if (DEPTH == 256 && MACRO_W == 96) begin : m256x96
                fakeram45_256x96 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({96{1'b1}}));
            end
            else if (DEPTH == 512 && MACRO_W == 64) begin : m512x64
                fakeram45_512x64 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({64{1'b1}}));
            end
            else if (DEPTH == 1024 && MACRO_W == 32) begin : m1024x32
                fakeram45_1024x32 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({32{1'b1}}));
            end
            else if (DEPTH == 2048 && MACRO_W == 39) begin : m2048x39
                fakeram45_2048x39 u_ram (.clk(clk), .rd_out(rd), .ce_in(ce), .we_in(we),
                    .addr_in(addr), .wd_in(wd), .w_mask_in({39{1'b1}}));
            end
        end
    endgenerate
endmodule
