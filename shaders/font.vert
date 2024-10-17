#version 460 core
layout(location=0) in vec3 inPos;
layout(location=1) in vec4 inColor;
layout(location=2) in vec2 inTexCoords;
layout (location=0) out vec2 aTexCoords;
layout (location=1) out vec4 aColor;



layout(set =0, binding =0) uniform ProjMatrix{
    mat4 proj;
}projMatrix;

void main(){
	gl_Position = projMatrix.proj * vec4(inPos,1.0);
	aTexCoords = inTexCoords;
	aColor = inColor;
}