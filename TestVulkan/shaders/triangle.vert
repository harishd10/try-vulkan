#version 450

layout (location = 0) in vec2 inPos;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

void main() 
{
    gl_Position = vec4(inPos.xy, 1.0, 1.0);
    gl_PointSize = 1;
}
