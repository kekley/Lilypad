#version 450
layout (location = 0)in vec2 inPos; //vec 2 position, vec2 texCoords
layout (location = 1)in vec2 uv;
layout (location = 2)out vec2 texCoords;
layout(location = 3 ) out vec4 outColor;
layout(location = 4) flat out uint tex_index;

layout(push_constant) uniform constants{
    mat4 model_matrix;
    vec4 color;
    uint index;
}PushConstants;

layout(set =0, binding =0) uniform ProjMatrix{
    mat4 proj;
}projMatrix;

void main(){
    texCoords = uv;
    tex_index = PushConstants.index;
    outColor = PushConstants.color;
    gl_Position = projMatrix.proj * PushConstants.model_matrix*vec4(inPos,0.0,1.0);
} 