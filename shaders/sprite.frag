#version 460 core
layout(location = 0)out vec4 outColor;
layout(location = 2)in vec2 texCoords;
layout(location = 3) in vec4 inColor;
layout(location = 4) flat in uint tex_index;

layout(push_constant) uniform constants{
    mat4 model_matrix;
    vec4 color;
    uint index;
}PushConstants;

layout(set = 1,binding =0 ) uniform sampler sampl;
layout(set=1, binding = 1) uniform texture2D textures[512];

void main()
{    
    outColor = texture(sampler2D(textures[tex_index],sampl),texCoords)*inColor;
    //outColor = vec4(texCoords,0.0,1.0)*inColor;
}  