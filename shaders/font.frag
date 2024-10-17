#version 460 core
layout (location=0) in vec2 aTexCoords;
layout (location=1) in vec4 aColor;

layout (location=2) out vec4 outFragColor;

layout(set =1,binding = 0) uniform sampler2D spriteTexture;

void main() {
	vec4 sampled = vec4(1.0,1.0,1.0,texture(spriteTexture,aTexCoords).r);
	outFragColor = aColor * sampled;
}