#version 450


layout(location = 0) out vec4 f_color;
layout(location = 3) in vec4 inColor;

void main() {
    f_color = inColor;
}
