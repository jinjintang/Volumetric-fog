#version 450


layout (binding = 1) uniform sampler2D samplerColor;
layout (binding = 2) uniform sampler2D samplerDepth;
layout (binding = 3) uniform sampler3D _VolumeScatter;
layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

vec4 _VolumeScatter_TexelSize=vec4(1.0f / 128, 1.0f / 128, 1.0f / 128, 0);
vec4 _Screen_TexelSize=vec4(1.0f / 1024, 1.0f / 1024, 1024, 1024);
float _CameraFarOverMaxFar=1.0;
float _NearOverFarClip=0.1/128;

float LinearizeDepth(float depth)
{
  float n = 1.0; // camera z near
  float f = 128.0; // camera z far
  float z = depth;
 // return (2.0 * n) / (f + n - z * (f - n));	
	return ( n * f) / (-f+ depth * (f - n))/f;	

}

int ihash(int n)
{
	n = (n<<13)^n;
	return (n*(n*n*15731+789221)+1376312589) & 2147483647;
}

float frand(int n)
{
	return ihash(n) / 2147483647.0;
}

vec2 cellNoise(ivec2 p)
{
	int i = p.y*256 + p.x;
	return vec2(frand(i), frand(i + 57)) - 0.5;//*2.0-1.0;
}
vec4 Fog(float linear01Depth, vec2 screenuv)
{
	float z = linear01Depth * _CameraFarOverMaxFar;
	z = (z - _NearOverFarClip) / (1 - _NearOverFarClip);
	//if (z < 0.0)
	//	return vec4(0, 0, 0, 1);

	vec3 uvw = vec3(screenuv.x, screenuv.y, 0.5);
	uvw.xy += cellNoise(ivec2(uvw.xy * _Screen_TexelSize.zw)) * _VolumeScatter_TexelSize.xy * 0.8;
	return texture(_VolumeScatter,uvw);
}

void main() 
{
	

	
	float depth = texture(samplerDepth, inUV).r;
	vec4 color = texture(samplerColor, inUV);

	vec4 fog = Fog(LinearizeDepth(depth), inUV);
	outFragColor = color*fog.a+fog;
	
	
}