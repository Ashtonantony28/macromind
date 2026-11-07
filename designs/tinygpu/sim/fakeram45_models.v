// MacroMind: behavioural simulation models of the NanGate45 fakeram45 macros
// (1RW, registered read, per-bit write mask). Simulation only; ORFS uses the
// platform .lib/.lef blackboxes.
`timescale 1ns/1ns
module fakeram45_32x32 (input wire clk, input wire ce_in, input wire we_in,
    input wire [4:0] addr_in, input wire [31:0] wd_in, input wire [31:0] w_mask_in,
    output reg [31:0] rd_out);
    reg [31:0] mem [0:31];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_32x64 (input wire clk, input wire ce_in, input wire we_in,
    input wire [4:0] addr_in, input wire [63:0] wd_in, input wire [63:0] w_mask_in,
    output reg [63:0] rd_out);
    reg [63:0] mem [0:31];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_64x7 (input wire clk, input wire ce_in, input wire we_in,
    input wire [5:0] addr_in, input wire [6:0] wd_in, input wire [6:0] w_mask_in,
    output reg [6:0] rd_out);
    reg [6:0] mem [0:63];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_64x15 (input wire clk, input wire ce_in, input wire we_in,
    input wire [5:0] addr_in, input wire [14:0] wd_in, input wire [14:0] w_mask_in,
    output reg [14:0] rd_out);
    reg [14:0] mem [0:63];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_64x21 (input wire clk, input wire ce_in, input wire we_in,
    input wire [5:0] addr_in, input wire [20:0] wd_in, input wire [20:0] w_mask_in,
    output reg [20:0] rd_out);
    reg [20:0] mem [0:63];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_64x32 (input wire clk, input wire ce_in, input wire we_in,
    input wire [5:0] addr_in, input wire [31:0] wd_in, input wire [31:0] w_mask_in,
    output reg [31:0] rd_out);
    reg [31:0] mem [0:63];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_64x62 (input wire clk, input wire ce_in, input wire we_in,
    input wire [5:0] addr_in, input wire [61:0] wd_in, input wire [61:0] w_mask_in,
    output reg [61:0] rd_out);
    reg [61:0] mem [0:63];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_64x64 (input wire clk, input wire ce_in, input wire we_in,
    input wire [5:0] addr_in, input wire [63:0] wd_in, input wire [63:0] w_mask_in,
    output reg [63:0] rd_out);
    reg [63:0] mem [0:63];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_64x96 (input wire clk, input wire ce_in, input wire we_in,
    input wire [5:0] addr_in, input wire [95:0] wd_in, input wire [95:0] w_mask_in,
    output reg [95:0] rd_out);
    reg [95:0] mem [0:63];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_64x124 (input wire clk, input wire ce_in, input wire we_in,
    input wire [5:0] addr_in, input wire [123:0] wd_in, input wire [123:0] w_mask_in,
    output reg [123:0] rd_out);
    reg [123:0] mem [0:63];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_128x32 (input wire clk, input wire ce_in, input wire we_in,
    input wire [6:0] addr_in, input wire [31:0] wd_in, input wire [31:0] w_mask_in,
    output reg [31:0] rd_out);
    reg [31:0] mem [0:127];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_128x116 (input wire clk, input wire ce_in, input wire we_in,
    input wire [6:0] addr_in, input wire [115:0] wd_in, input wire [115:0] w_mask_in,
    output reg [115:0] rd_out);
    reg [115:0] mem [0:127];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_128x256 (input wire clk, input wire ce_in, input wire we_in,
    input wire [6:0] addr_in, input wire [255:0] wd_in, input wire [255:0] w_mask_in,
    output reg [255:0] rd_out);
    reg [255:0] mem [0:127];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_256x16 (input wire clk, input wire ce_in, input wire we_in,
    input wire [7:0] addr_in, input wire [15:0] wd_in, input wire [15:0] w_mask_in,
    output reg [15:0] rd_out);
    reg [15:0] mem [0:255];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_256x32 (input wire clk, input wire ce_in, input wire we_in,
    input wire [7:0] addr_in, input wire [31:0] wd_in, input wire [31:0] w_mask_in,
    output reg [31:0] rd_out);
    reg [31:0] mem [0:255];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_256x34 (input wire clk, input wire ce_in, input wire we_in,
    input wire [7:0] addr_in, input wire [33:0] wd_in, input wire [33:0] w_mask_in,
    output reg [33:0] rd_out);
    reg [33:0] mem [0:255];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_256x48 (input wire clk, input wire ce_in, input wire we_in,
    input wire [7:0] addr_in, input wire [47:0] wd_in, input wire [47:0] w_mask_in,
    output reg [47:0] rd_out);
    reg [47:0] mem [0:255];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_256x95 (input wire clk, input wire ce_in, input wire we_in,
    input wire [7:0] addr_in, input wire [94:0] wd_in, input wire [94:0] w_mask_in,
    output reg [94:0] rd_out);
    reg [94:0] mem [0:255];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_256x96 (input wire clk, input wire ce_in, input wire we_in,
    input wire [7:0] addr_in, input wire [95:0] wd_in, input wire [95:0] w_mask_in,
    output reg [95:0] rd_out);
    reg [95:0] mem [0:255];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_512x64 (input wire clk, input wire ce_in, input wire we_in,
    input wire [8:0] addr_in, input wire [63:0] wd_in, input wire [63:0] w_mask_in,
    output reg [63:0] rd_out);
    reg [63:0] mem [0:511];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_1024x32 (input wire clk, input wire ce_in, input wire we_in,
    input wire [9:0] addr_in, input wire [31:0] wd_in, input wire [31:0] w_mask_in,
    output reg [31:0] rd_out);
    reg [31:0] mem [0:1023];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
module fakeram45_2048x39 (input wire clk, input wire ce_in, input wire we_in,
    input wire [10:0] addr_in, input wire [38:0] wd_in, input wire [38:0] w_mask_in,
    output reg [38:0] rd_out);
    reg [38:0] mem [0:2047];
    always @(posedge clk) if (ce_in) begin
        if (we_in) mem[addr_in] <= (mem[addr_in] & ~w_mask_in) | (wd_in & w_mask_in);
        else rd_out <= mem[addr_in];
    end
endmodule
