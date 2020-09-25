/*
* Vulkan Example - Shadow mapping for directional light sources
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"
#include "VulkanBuffer.hpp"
#include "VulkanModel.hpp"
#include "VulkanTexture.hpp"
#define ENABLE_VALIDATION false

// 16 bits of depth is enough for such a small scene
#define DEPTH_FORMAT VK_FORMAT_D16_UNORM

// Shadowmap properties
#if defined(__ANDROID__)
#define SHADOWMAP_DIM 1024
#else
#define SHADOWMAP_DIM 2048
#endif
#define SHADOWMAP_FILTER VK_FILTER_LINEAR
#define TEX_DIM 512
using namespace std;
class VulkanExample : public VulkanExampleBase
{
public:
	float animationTimer = 0.0f;
	float m_ConstantFog = 100;
	float m_HeightFogAmount =0;
	float m_HeightFogExponent = 1;
	float m_HeightFogOffset = 0;
	VkCommandBuffer offScreenCmdBuffer = VK_NULL_HANDLE;
	VkSemaphore offscreenSemaphore;
	struct Texture {
		VkSampler sampler = VK_NULL_HANDLE;
		VkImage image = VK_NULL_HANDLE;
		VkImageLayout imageLayout;
		VkDeviceMemory deviceMemory = VK_NULL_HANDLE;
		VkImageView view = VK_NULL_HANDLE;
		VkDescriptorImageInfo descriptor;
		VkFormat format;
		uint32_t width, height, depth;
		uint32_t mipLevels;
		vks::VulkanDevice *device;
	} texture;
	Texture VolumeInject;
	Texture VolumeScatter;
	struct {
		uint32_t queueFamilyIndex;
		VkSemaphore semaphore;
	
	}graphics;
	struct {
		uint32_t queueFamilyIndex;
		VkQueue queue;
		VkSemaphore semaphore[2];
		VkDescriptorSetLayout descriptorSetLayout[2];
		VkDescriptorSet descriptorSet[2];
		VkPipelineLayout pipelineLayout[2];
		VkPipeline pipeline[2];

		VkCommandPool commandPool;
		VkCommandBuffer commandBuffer[2];
		VkFence fence;
		struct UBOCompute {							// Compute shader uniform block object
			 glm::vec4 _FrustumRays[4];
			 glm::vec4 _CameraPos;
			 glm::vec4 _FogParams;
			 glm::vec3 _AmbientLight;
			 float _Density;
			 glm::vec3 _DirLightColor;
			 float _Intensity;
			 glm::vec3 _DirLightDir;
			 float _Anisotropy;
			 float _NearOverFarClip;
			 float _Time;
		} ubocompute;
		vks::Buffer uniformBuffer;
	} compute;

	bool displayShadowMap = false;
	bool filterPCF = true;

	float farClip = 128.0;
	float nearClip = 0.1f;
	// Keep depth range as small as possible
	// for better shadow map precision
	float zNear = 1.0f;
	float zFar = 96.0f;

	// Depth bias (and slope) are used to avoid sadowing artefacts
	// Constant depth bias factor (always applied)
	float depthBiasConstant = 1.25f;
	// Slope depth bias factor, applied depending on polygon's slope
	float depthBiasSlope = 1.75f;

	glm::vec3 lightPos = glm::vec3();
	float lightFOV = 45.0f;
	
	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_UV,
		vks::VERTEX_COMPONENT_COLOR,
		vks::VERTEX_COMPONENT_NORMAL,
	});

	struct {
		vks::Model quad;
	} models;

	std::vector<vks::Model> scenes;
	std::vector<std::string> sceneNames;
	int32_t sceneIndex = 0;

	struct {
		vks::Buffer scene;
		vks::Buffer offscreen;
		vks::Buffer debug;
	} uniformBuffers;

	struct {
		glm::mat4 projection;
		glm::mat4 model;
	} uboVSquad;

	struct {
		glm::mat4 projection;
		glm::mat4 view;
		glm::mat4 model;
		glm::mat4 depthBiasMVP;
		glm::vec3 lightPos;
	} uboVSscene;

	struct {
		glm::mat4 depthMVP;
	} uboOffscreenVS;

	struct {
		VkPipeline quad;
		VkPipeline offscreen;
		VkPipeline sceneShadow;
		VkPipeline sceneShadowPCF;
	} pipelines;

	struct {
		VkPipelineLayout quad;
		VkPipelineLayout offscreen;
		VkPipelineLayout scene;
	} pipelineLayouts;

	struct {
		VkDescriptorSet offscreen;
		VkDescriptorSet scene;
	} descriptorSets;

	VkDescriptorSet descriptorSet;
	VkDescriptorSetLayout descriptorSetLayout;

	// Framebuffer for offscreen rendering
	struct FrameBufferAttachment {
		VkImage image;
		VkDeviceMemory mem;
		VkImageView view;
	};
	struct OffscreenPass {
		int32_t width, height;
		VkFramebuffer frameBuffer;
		FrameBufferAttachment depth;
		VkRenderPass renderPass;
		VkSampler depthSampler;
		VkDescriptorImageInfo descriptor;
	} offscreenPass;

	
	struct ScenePass {
		int32_t width, height;
		VkFramebuffer frameBuffer;
		FrameBufferAttachment color,depth;
		VkRenderPass renderPass;
		VkSampler colorSampler,depthSampler;
		VkDescriptorImageInfo colordescriptor,depthdescriptor;
	} scenePass;

	;


	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Projected shadow mapping";
		camera.type = Camera::CameraType::lookat;
		camera.setPosition(glm::vec3(0.0f, -0.0f, -20.0f));
		//camera.setRotation(glm::vec3(-15.0f, -390.0f, 0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 1.0f, 256.0f);
		

		
		timerSpeed *= 0.5f;
		settings.overlay = true;
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class
		vkDeviceWaitIdle(device);
		// Frame buffer
		vkDestroySampler(device, offscreenPass.depthSampler, nullptr);
		vkDestroySampler(device, VolumeInject.sampler, nullptr);
		vkDestroySampler(device, VolumeScatter.sampler, nullptr);

		// Depth attachment
		vkDestroyImageView(device, offscreenPass.depth.view, nullptr);
		vkDestroyImageView(device, VolumeInject.view, nullptr);
		vkDestroyImageView(device, VolumeScatter.view, nullptr);

		vkDestroyImage(device, offscreenPass.depth.image, nullptr);
		vkDestroyImage(device, VolumeInject.image, nullptr);
		vkDestroyImage(device, VolumeScatter.image, nullptr);


		vkFreeMemory(device, offscreenPass.depth.mem, nullptr);
		vkFreeMemory(device, VolumeInject.deviceMemory, nullptr);
		vkFreeMemory(device, VolumeScatter.deviceMemory, nullptr);

		vkDestroyFramebuffer(device, offscreenPass.frameBuffer, nullptr);

		vkDestroyRenderPass(device, offscreenPass.renderPass, nullptr);

		vkDestroyPipeline(device, pipelines.quad, nullptr);
		vkDestroyPipeline(device, pipelines.offscreen, nullptr);
		vkDestroyPipeline(device, pipelines.sceneShadow, nullptr);
		vkDestroyPipeline(device, pipelines.sceneShadowPCF, nullptr);
		vkDestroyPipeline(device, compute.pipeline[0], nullptr);
		vkDestroyPipeline(device, compute.pipeline[1], nullptr);


		vkDestroyPipelineLayout(device, pipelineLayouts.quad, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayouts.offscreen, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayouts.scene, nullptr);
		vkDestroyPipelineLayout(device, compute.pipelineLayout[0], nullptr);
		vkDestroyPipelineLayout(device, compute.pipelineLayout[1], nullptr);


		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout[0], nullptr);
		vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout[1], nullptr);
		// Meshes
		for (auto scene : scenes) {
			scene.destroy();
		}
		models.quad.destroy();

		vkDestroySemaphore(device,offscreenSemaphore, nullptr);

		vkDestroySemaphore(device, compute.semaphore[0], nullptr);
		vkDestroySemaphore(device, compute.semaphore[1], nullptr);

		// Uniform buffers
		uniformBuffers.offscreen.destroy();
		uniformBuffers.scene.destroy();
		uniformBuffers.debug.destroy();
		compute.uniformBuffer.destroy();
		
		vkDestroyCommandPool(device, compute.commandPool, nullptr);
		
	}

	// Prepare a texture target that is used to store compute shader calculations
	void prepareTextureTarget(Texture *tex, uint32_t width, uint32_t height, uint32_t depth, VkFormat format)
	{
		// Get device properties for the requested texture format
		VkFormatProperties formatProperties;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProperties);
		// Check if requested image format supports image storage operations
		assert(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);

		// Prepare blit target texture
		tex->width = width;
		tex->height = height;
		tex->depth = depth;
		VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
		imageCreateInfo.imageType = VK_IMAGE_TYPE_3D;
		imageCreateInfo.format = format;
		imageCreateInfo.extent = { width, height, depth };
		imageCreateInfo.mipLevels = 1;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		// Image will be sampled in the fragment shader and used as storage target in the compute shader
		imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		imageCreateInfo.flags = 0;

		VkMemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;

		VK_CHECK_RESULT(vkCreateImage(device, &imageCreateInfo, nullptr, &tex->image));
		vkGetImageMemoryRequirements(device, tex->image, &memReqs);
		memAllocInfo.allocationSize = memReqs.size;
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr, &tex->deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, tex->image, tex->deviceMemory, 0));

		VkCommandBuffer layoutCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

		tex->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		vks::tools::setImageLayout(
			layoutCmd,
			tex->image,
			VK_IMAGE_ASPECT_COLOR_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			tex->imageLayout);

		vulkanDevice->flushCommandBuffer(layoutCmd, queue, true);

		// Create sampler
		VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		
		sampler.magFilter = VK_FILTER_LINEAR;
		sampler.minFilter = VK_FILTER_LINEAR;
		sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.maxAnisotropy = 1.0f;
		sampler.compareOp = VK_COMPARE_OP_NEVER;
		sampler.minLod = 0.0f;
		sampler.maxLod = 0.0f;
		sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &tex->sampler));

		// Create image view
		VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
		view.viewType = VK_IMAGE_VIEW_TYPE_3D;
		view.format = format;
		view.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
		view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		view.image = tex->image;
		VK_CHECK_RESULT(vkCreateImageView(device, &view, nullptr, &tex->view));

		// Initialize a descriptor for later use
		tex->descriptor.imageLayout = tex->imageLayout;
		tex->descriptor.imageView = tex->view;
		tex->descriptor.sampler = tex->sampler;
	//	tex->device = vulkanDevice;
	}

	// Set up a separate render pass for the offscreen frame buffer
	// This is necessary as the offscreen frame buffer attachments use formats different to those from the example render pass
	void prepareOffscreenRenderpass()
	{
		VkAttachmentDescription attachmentDescription{};
		attachmentDescription.format = DEPTH_FORMAT;
		attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
		attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;							// Clear depth at beginning of the render pass
		attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;						// We will read from depth, so it's important to store the depth attachment results
		attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;					// We don't care about initial layout of the attachment
		attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;// Attachment will be transitioned to shader read at render pass end

		VkAttachmentReference depthReference = {};
		depthReference.attachment = 0;
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;			// Attachment will be used as depth/stencil during render pass

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 0;													// No color attachments
		subpass.pDepthStencilAttachment = &depthReference;									// Reference to our depth attachment

		// Use subpass dependencies for layout transitions
		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo renderPassCreateInfo = vks::initializers::renderPassCreateInfo();
		renderPassCreateInfo.attachmentCount = 1;
		renderPassCreateInfo.pAttachments = &attachmentDescription;
		renderPassCreateInfo.subpassCount = 1;
		renderPassCreateInfo.pSubpasses = &subpass;
		renderPassCreateInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassCreateInfo.pDependencies = dependencies.data();

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &offscreenPass.renderPass));
	}
	

	// Setup the offscreen framebuffer for rendering the scene from light's point-of-view to
	// The depth attachment of this framebuffer will then be used to sample from in the fragment shader of the shadowing pass
	void prepareOffscreenFramebuffer()
	{
		offscreenPass.width = SHADOWMAP_DIM;
		offscreenPass.height = SHADOWMAP_DIM;

		// For shadow mapping we only need a depth attachment
		VkImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = VK_IMAGE_TYPE_2D;
		image.extent.width = offscreenPass.width;
		image.extent.height = offscreenPass.height;
		image.extent.depth = 1;
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = VK_SAMPLE_COUNT_1_BIT;
		image.tiling = VK_IMAGE_TILING_OPTIMAL;
		image.format = DEPTH_FORMAT;																// Depth stencil attachment
		image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;		// We will sample directly from the depth attachment for the shadow mapping
		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &offscreenPass.depth.image));

		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device, offscreenPass.depth.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &offscreenPass.depth.mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, offscreenPass.depth.image, offscreenPass.depth.mem, 0));

		VkImageViewCreateInfo depthStencilView = vks::initializers::imageViewCreateInfo();
		depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		depthStencilView.format = DEPTH_FORMAT;
		depthStencilView.subresourceRange = {};
		depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		depthStencilView.subresourceRange.baseMipLevel = 0;
		depthStencilView.subresourceRange.levelCount = 1;
		depthStencilView.subresourceRange.baseArrayLayer = 0;
		depthStencilView.subresourceRange.layerCount = 1;
		depthStencilView.image = offscreenPass.depth.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &depthStencilView, nullptr, &offscreenPass.depth.view));

		// Create sampler to sample from to depth attachment 
		// Used to sample in the fragment shader for shadowed rendering
		VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = SHADOWMAP_FILTER;
		sampler.minFilter = SHADOWMAP_FILTER;
		sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.maxAnisotropy = 1.0f;
		sampler.minLod = 0.0f;
		sampler.maxLod = 1.0f;
		sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &offscreenPass.depthSampler));

		prepareOffscreenRenderpass();

		// Create frame buffer
		VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
		fbufCreateInfo.renderPass = offscreenPass.renderPass; 
		fbufCreateInfo.attachmentCount = 1;
		fbufCreateInfo.pAttachments = &offscreenPass.depth.view;
		fbufCreateInfo.width = offscreenPass.width;
		fbufCreateInfo.height = offscreenPass.height;
		fbufCreateInfo.layers = 1;

		VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &offscreenPass.frameBuffer));
	}
	// Set up a separate render pass for the offscreen frame buffer
	// This is necessary as the offscreen frame buffer attachments use formats different to those from the example render pass
	void prepareSceneRenderpass()
	{
		scenePass.width = 1024;
		scenePass.height = 1024;

		// Find a suitable depth format
		VkFormat fbDepthFormat;
		VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &fbDepthFormat);
		assert(validDepthFormat);

		// Create a separate render pass for the scene rendering as it may differ from the one used for scene rendering

		std::array<VkAttachmentDescription, 2> attchmentDescriptions = {};
		// Color attachment
		attchmentDescriptions[0].format = VK_FORMAT_R8G8B8A8_UNORM;
		attchmentDescriptions[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attchmentDescriptions[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attchmentDescriptions[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attchmentDescriptions[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attchmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attchmentDescriptions[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attchmentDescriptions[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		// Depth attachment
		attchmentDescriptions[1].format = fbDepthFormat;
		attchmentDescriptions[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attchmentDescriptions[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attchmentDescriptions[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attchmentDescriptions[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attchmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attchmentDescriptions[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attchmentDescriptions[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
		VkAttachmentReference depthReference = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

		VkSubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDescription.colorAttachmentCount = 1;
		subpassDescription.pColorAttachments = &colorReference;
		subpassDescription.pDepthStencilAttachment = &depthReference;

		// Use subpass dependencies for layout transitions
		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		// Create the actual renderpass
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attchmentDescriptions.size());
		renderPassInfo.pAttachments = attchmentDescriptions.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDescription;
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &scenePass.renderPass));

		// Create sampler to sample from the color attachments
		VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = VK_FILTER_LINEAR;
		sampler.minFilter = VK_FILTER_LINEAR;
		sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.maxAnisotropy = 1.0f;
		sampler.minLod = 0.0f;
		sampler.maxLod = 1.0f;
		sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &scenePass.colorSampler));
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &scenePass.depthSampler));

		// Create two frame buffers
		prepareSceneFramebuffer(VK_FORMAT_R8G8B8A8_UNORM, fbDepthFormat);
		
	}
	// Setup the offscreen framebuffer for rendering the scene from light's point-of-view to
	// The depth attachment of this framebuffer will then be used to sample from in the fragment shader of the shadowing pass
	void prepareSceneFramebuffer( VkFormat colorFormat, VkFormat depthFormat)
	{
		// Color attachment
		VkImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = VK_IMAGE_TYPE_2D;
		image.format = colorFormat;
		image.extent.width = 1024;
		image.extent.height = 1024;
		image.extent.depth = 1;
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = VK_SAMPLE_COUNT_1_BIT;
		image.tiling = VK_IMAGE_TILING_OPTIMAL;
		// We will sample directly from the color attachment
		image.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;

		VkImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
		colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		colorImageView.format = colorFormat;
		colorImageView.flags = 0;
		colorImageView.subresourceRange = {};
		colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		colorImageView.subresourceRange.baseMipLevel = 0;
		colorImageView.subresourceRange.levelCount = 1;
		colorImageView.subresourceRange.baseArrayLayer = 0;
		colorImageView.subresourceRange.layerCount = 1;

		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &scenePass.color.image));
		vkGetImageMemoryRequirements(device, scenePass.color.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &scenePass.color.mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, scenePass.color.image, scenePass.color.mem, 0));

		colorImageView.image = scenePass.color.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &colorImageView, nullptr, &scenePass.color.view));

		// Depth stencil attachment
		image.format = depthFormat;
		image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

		VkImageViewCreateInfo depthStencilView = vks::initializers::imageViewCreateInfo();
		depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		depthStencilView.format = depthFormat;
		depthStencilView.flags = 0;
		depthStencilView.subresourceRange = {};
		depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
		depthStencilView.subresourceRange.baseMipLevel = 0;
		depthStencilView.subresourceRange.levelCount = 1;
		depthStencilView.subresourceRange.baseArrayLayer = 0;
		depthStencilView.subresourceRange.layerCount = 1;

		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &scenePass.depth.image));
		vkGetImageMemoryRequirements(device, scenePass.depth.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &scenePass.depth.mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, scenePass.depth.image, scenePass.depth.mem, 0));

		depthStencilView.image = scenePass.depth.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &depthStencilView, nullptr, &scenePass.depth.view));

		VkImageView attachments[2];
		attachments[0] = scenePass.color.view;
		attachments[1] = scenePass.depth.view;

		VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
		fbufCreateInfo.renderPass = scenePass.renderPass;
		fbufCreateInfo.attachmentCount = 2;
		fbufCreateInfo.pAttachments = attachments;
		fbufCreateInfo.width = 1024;
		fbufCreateInfo.height = 1024;
		fbufCreateInfo.layers = 1;

		VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &scenePass.frameBuffer));

		
	}

	void buildDeferredCommandBuffer(bool rebuild = false)
	{
		if ((offScreenCmdBuffer == VK_NULL_HANDLE) || (rebuild))
		{
			if (rebuild)
			{
				vkFreeCommandBuffers(device, cmdPool, 1, &offScreenCmdBuffer);
			}
			// Create a command buffer for compute operations
			VkCommandBufferAllocateInfo cmdBufAllocateInfo =
				vks::initializers::commandBufferAllocateInfo(
					cmdPool,
					VK_COMMAND_BUFFER_LEVEL_PRIMARY,
					1);

			VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &offScreenCmdBuffer));

		}

		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &offscreenSemaphore));

		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		VkViewport viewport;
		VkRect2D scissor;
		VkDeviceSize offsets[1] = { 0 };
		VK_CHECK_RESULT(vkBeginCommandBuffer(offScreenCmdBuffer, &cmdBufInfo));

		{

			clearValues[0].depthStencil = { 1.0f, 0 };

			VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
			renderPassBeginInfo.renderPass = offscreenPass.renderPass;
			renderPassBeginInfo.framebuffer = offscreenPass.frameBuffer;
			renderPassBeginInfo.renderArea.extent.width = offscreenPass.width;
			renderPassBeginInfo.renderArea.extent.height = offscreenPass.height;
			renderPassBeginInfo.clearValueCount = 1;
			renderPassBeginInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(offScreenCmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			viewport = vks::initializers::viewport((float)offscreenPass.width, (float)offscreenPass.height, 0.0f, 1.0f);
			vkCmdSetViewport(offScreenCmdBuffer, 0, 1, &viewport);

			scissor = vks::initializers::rect2D(offscreenPass.width, offscreenPass.height, 0, 0);
			vkCmdSetScissor(offScreenCmdBuffer, 0, 1, &scissor);

			// Set depth bias (aka "Polygon offset")
			// Required to avoid shadow mapping artefacts
			vkCmdSetDepthBias(
				offScreenCmdBuffer,
				depthBiasConstant,
				0.0f,
				depthBiasSlope);

			vkCmdBindPipeline(offScreenCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.offscreen);
			vkCmdBindDescriptorSets(offScreenCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.offscreen, 0, 1, &descriptorSets.offscreen, 0, NULL);

			vkCmdBindVertexBuffers(offScreenCmdBuffer, 0, 1, &scenes[sceneIndex].vertices.buffer, offsets);
			vkCmdBindIndexBuffer(offScreenCmdBuffer, scenes[sceneIndex].indices.buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(offScreenCmdBuffer, scenes[sceneIndex].indexCount, 1, 0, 0, 0);

			vkCmdEndRenderPass(offScreenCmdBuffer);
		}
		VK_CHECK_RESULT(vkEndCommandBuffer(offScreenCmdBuffer));
	}

	void buildCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		VkViewport viewport;
		VkRect2D scissor;
		VkDeviceSize offsets[1] = { 0 };

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			{

				clearValues[0].color = defaultClearColor;
				clearValues[1].depthStencil = { 1.0f, 0 };


				VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
				renderPassBeginInfo.renderPass = scenePass.renderPass;
				renderPassBeginInfo.framebuffer = scenePass.frameBuffer;
				renderPassBeginInfo.renderArea.extent.width = scenePass.width;
				renderPassBeginInfo.renderArea.extent.height = scenePass.height;
				renderPassBeginInfo.clearValueCount = 2;
				renderPassBeginInfo.pClearValues = clearValues;

				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

				viewport = vks::initializers::viewport((float)scenePass.width, (float)scenePass.height, 0.0f, 1.0f);
				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

				scissor = vks::initializers::rect2D(offscreenPass.width, offscreenPass.height, 0, 0);
				vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

				// 3D scene
				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.scene, 0, 1, &descriptorSets.scene, 0, NULL);
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, (filterPCF) ? pipelines.sceneShadowPCF : pipelines.sceneShadow);

				vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &scenes[sceneIndex].vertices.buffer, offsets);
				vkCmdBindIndexBuffer(drawCmdBuffers[i], scenes[sceneIndex].indices.buffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdDrawIndexed(drawCmdBuffers[i], scenes[sceneIndex].indexCount, 1, 0, 0, 0);

			}
			{
				clearValues[0].color = defaultClearColor;
				clearValues[1].depthStencil = { 1.0f, 0 };

				VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
				renderPassBeginInfo.renderPass = renderPass;
				renderPassBeginInfo.framebuffer = frameBuffers[i];
				renderPassBeginInfo.renderArea.extent.width = width;
				renderPassBeginInfo.renderArea.extent.height = height;
				renderPassBeginInfo.clearValueCount = 2;
				renderPassBeginInfo.pClearValues = clearValues;

				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

				viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

				scissor = vks::initializers::rect2D(width, height, 0, 0);
				vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);


				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.quad, 0, 1, &descriptorSet, 0, NULL);
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.quad);
				vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &models.quad.vertices.buffer, offsets);
				vkCmdBindIndexBuffer(drawCmdBuffers[i], models.quad.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdDrawIndexed(drawCmdBuffers[i], models.quad.indexCount, 1, 0, 0, 0);



				//drawUI(drawCmdBuffers[i]);

				VkImageMemoryBarrier  imageMemoryBarrier = {};
				imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
				imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
				imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
				imageMemoryBarrier.image = VolumeScatter.image;
				imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
				imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
				imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				imageMemoryBarrier.srcQueueFamilyIndex = graphics.queueFamilyIndex;
				imageMemoryBarrier.srcQueueFamilyIndex = compute.queueFamilyIndex;

				vkCmdPipelineBarrier(
					drawCmdBuffers[i],
					VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_FLAGS_NONE,
					0, nullptr,
					0, nullptr,
					1, &imageMemoryBarrier);
				vkCmdEndRenderPass(drawCmdBuffers[i]);
			}

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void loadAssets()
	{
		scenes.resize(2);
		scenes[0].loadFromFile(getAssetPath() + "models/vulkanscene_shadow.dae", vertexLayout, 4.0f, vulkanDevice, queue);
		scenes[1].loadFromFile(getAssetPath() + "models/samplescene.dae", vertexLayout, 0.25f, vulkanDevice, queue);
		sceneNames = {"Vulkan scene", "Teapots and pillars" };
	}

	void generateQuad()
	{
		// Setup vertices for a single uv-mapped quad
		struct Vertex {
			float pos[3];
			float uv[2];
			float col[3];
			float normal[3];
		};

#define QUAD_COLOR_NORMAL { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 1.0f }
		std::vector<Vertex> vertexBuffer =
		{
			{ { 1.0f, 1.0f, 0.0f },{ 1.0f, 1.0f }, QUAD_COLOR_NORMAL },
			{ { 0.0f, 1.0f, 0.0f },{ 0.0f, 1.0f }, QUAD_COLOR_NORMAL },
			{ { 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f }, QUAD_COLOR_NORMAL },
			{ { 1.0f, 0.0f, 0.0f },{ 1.0f, 0.0f }, QUAD_COLOR_NORMAL }
		};
#undef QUAD_COLOR_NORMAL

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			vertexBuffer.size() * sizeof(Vertex),
			&models.quad.vertices.buffer,
			&models.quad.vertices.memory,
			vertexBuffer.data()));

		// Setup indices
		std::vector<uint32_t> indexBuffer = { 0,1,2, 2,3,0 };
		models.quad.indexCount = indexBuffer.size();

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			indexBuffer.size() * sizeof(uint32_t),
			&models.quad.indices.buffer,
			&models.quad.indices.memory,
			indexBuffer.data()));

		models.quad.device = device;
	}

	void setupDescriptorPool()
	{
		// Example uses three ubos and two image samplers
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 18),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,18)
		};

		VkDescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				poolSizes.size(),
				poolSizes.data(),
				5);

		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
	}

	void setupDescriptorSetLayout()
	{
		// Textured quad pipeline layout
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
			// Binding 1 : Fragment shader image sampler
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3)

		};

		VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

		VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&descriptorSetLayout,
				1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &pipelineLayouts.quad));

		// Offscreen pipeline layout
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &pipelineLayouts.offscreen));

		
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &pipelineLayouts.scene));


	}

	void setupDescriptorSets()
	{
		std::vector<VkWriteDescriptorSet> writeDescriptorSets;

		// Textured quad descriptor set
		VkDescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayout,
				1);

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

		// Image descriptor for the shadow map attachment
		VkDescriptorImageInfo texDescriptor =
			vks::initializers::descriptorImageInfo(
				offscreenPass.depthSampler,
				offscreenPass.depth.view,
				VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
		// Image descriptor for the shadow map attachment
		VkDescriptorImageInfo sceneDepthDescriptor =
			vks::initializers::descriptorImageInfo(
				scenePass.depthSampler,
				scenePass.depth.view,
				VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
		// Image descriptor for the shadow map attachment
		VkDescriptorImageInfo sceneColorDescriptor =
			vks::initializers::descriptorImageInfo(
				scenePass.colorSampler,
				scenePass.color.view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		writeDescriptorSets = {
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.debug.descriptor),
			// Binding 1 : Fragment shader texture sampler
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &sceneColorDescriptor),

			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, &sceneDepthDescriptor),

			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3, &VolumeScatter.descriptor)
		};
		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);

		// Offscreen
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.offscreen));

		writeDescriptorSets = {
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(descriptorSets.offscreen, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.offscreen.descriptor),
		};
		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);

		// 3D scene
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.scene));

		// Image descriptor for the shadow map attachment
		texDescriptor.sampler = offscreenPass.depthSampler;
		texDescriptor.imageView = offscreenPass.depth.view;

		writeDescriptorSets = {
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(descriptorSets.scene, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.scene.descriptor),
			// Binding 1 : Fragment shader shadow sampler
			vks::initializers::writeDescriptorSet(descriptorSets.scene, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &texDescriptor),

		};
		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);
	}

	void preparePipelines()
	{
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), dynamicStateEnables.size(), 0);
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayouts.quad, renderPass, 0);

		pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
		pipelineCI.pRasterizationState = &rasterizationStateCI;
		pipelineCI.pColorBlendState = &colorBlendStateCI;
		pipelineCI.pMultisampleState = &multisampleStateCI;
		pipelineCI.pViewportState = &viewportStateCI;
		pipelineCI.pDepthStencilState = &depthStencilStateCI;
		pipelineCI.pDynamicState = &dynamicStateCI;
		pipelineCI.stageCount = shaderStages.size();
		pipelineCI.pStages = shaderStages.data();

		// Shadow mapping debug quad display
		rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
		shaderStages[0] = loadShader(getAssetPath() + "shaders/shadowmapping/quad.vert", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/shadowmapping/quad.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
		// Empty vertex input state
		VkPipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		pipelineCI.pVertexInputState = &emptyInputState;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.quad));

		// Vertex bindings and attributes
		std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, vertexLayout.stride(), VK_VERTEX_INPUT_RATE_VERTEX),
		};
		std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0),				// Position			
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 3),	// Texture coordinates
			vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 5),	// Color
			vks::initializers::vertexInputAttributeDescription(0, 3, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 8),	// Normal
		};
		VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();
		pipelineCI.pVertexInputState = &vertexInputState;

		// Scene rendering with shadows applied
		rasterizationStateCI.cullMode = VK_CULL_MODE_BACK_BIT;
		shaderStages[0] = loadShader(getAssetPath() + "shaders/shadowmapping/scene.vert", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/shadowmapping/scene.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
		// Use specialization constants to select between horizontal and vertical blur
		uint32_t enablePCF = 0;
		VkSpecializationMapEntry specializationMapEntry = vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
		VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(1, &specializationMapEntry, sizeof(uint32_t), &enablePCF);
		shaderStages[1].pSpecializationInfo = &specializationInfo;
		// No filtering
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.sceneShadow));
		// PCF filtering
		enablePCF = 1;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.sceneShadowPCF));

		// Offscreen pipeline (vertex shader only)
		shaderStages[0] = loadShader(getAssetPath() + "shaders/shadowmapping/offscreen.vert", VK_SHADER_STAGE_VERTEX_BIT);
		pipelineCI.stageCount = 1;
		// No blend attachment states (no color attachments used)
		colorBlendStateCI.attachmentCount = 0;
		// Cull front faces
		depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		// Enable depth bias
		rasterizationStateCI.depthBiasEnable = VK_TRUE;
		// Add depth bias to dynamic state, so we can change it at runtime
		dynamicStateEnables.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
		dynamicStateCI =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables.data(),
				dynamicStateEnables.size(),
				0);

		pipelineCI.layout = pipelineLayouts.offscreen;
		pipelineCI.renderPass = offscreenPass.renderPass;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.offscreen));
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Debug quad vertex shader uniform buffer block
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniformBuffers.debug,
			sizeof(uboVSquad)));

		// Offscreen vertex shader uniform buffer block
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniformBuffers.offscreen,
			sizeof(uboOffscreenVS)));

		// Scene vertex shader uniform buffer block 
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniformBuffers.scene,
			sizeof(uboVSscene)));

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&compute.uniformBuffer,
			sizeof(compute.ubocompute));

		// Map persistent
		VK_CHECK_RESULT(uniformBuffers.debug.map());
		VK_CHECK_RESULT(uniformBuffers.offscreen.map());
		VK_CHECK_RESULT(uniformBuffers.scene.map());
		VK_CHECK_RESULT(compute.uniformBuffer.map());
		updateLight();
		updateUniformBufferOffscreen();
		updateUniformBuffers();
		updateComputeUniformBuffer();
	}
	glm::vec4 ViewportToWorldPoint(glm::vec3 pos) {
		pos.x = -pos.x*pos.z;
		pos.y = -pos.y*pos.z;
		float f = 256, n = 1;
		pos.z =  (-f * f / (f - n) - f  / (f - n));
		return glm::inverse(camera.matrices.view)*glm::inverse(camera.matrices.perspective)*glm::vec4(pos,-f);

	}
	void updateComputeUniformBuffer()
	{
		
		
		glm::vec4  cameraPos = glm::vec4(camera.position,1.0);
		glm::vec2 uvs[4] = {glm::vec2(-1,1),glm::vec2(1,1) ,glm::vec2(1,-1) ,glm::vec2(-1,-1) };
		
		for (int i = 0; i < 4; i++)
		{
			glm::vec3 ray = ViewportToWorldPoint(glm::vec3(uvs[i].x, uvs[i].y, 256))-cameraPos;
			compute.ubocompute._FrustumRays[i][0] = ray.x;
			compute.ubocompute._FrustumRays[i][1] = ray.y;
			compute.ubocompute._FrustumRays[i][2] = ray.z;
			compute.ubocompute._FrustumRays[i][3] = 1;
		}
		compute.ubocompute._CameraPos = cameraPos;
		float depthCompensation = (farClip - nearClip) * 0.01f;
		compute.ubocompute._Density = 1.0 * 0.128f * depthCompensation;
		compute.ubocompute._Intensity = 1.0f;
		compute.ubocompute._Anisotropy = 0.1;

		glm::vec4 m_fogParams;
		m_fogParams[0] = m_ConstantFog;
		m_fogParams[1] = m_HeightFogExponent;
		m_fogParams[2] = m_HeightFogOffset;
		m_fogParams[3] = m_HeightFogAmount;
		compute.ubocompute._FogParams=m_fogParams;
		compute.ubocompute._NearOverFarClip = nearClip / farClip;
		compute.ubocompute._Time = timer;
		compute.ubocompute._AmbientLight = glm::vec3(1,1,1);
		compute.ubocompute._DirLightColor = glm::vec3(1,1,0);
		compute.ubocompute._DirLightDir = lightPos;
		memcpy(compute.uniformBuffer.mapped, &compute.ubocompute, sizeof(compute.ubocompute));

	}

	void updateLight()
	{
		// Animate the light source
		lightPos.x = 50.0f; //cos(glm::radians(360.0f)) * 40.0f;
		lightPos.y = 0.0f ;
		lightPos.z =0.0f;
	}

	void updateUniformBuffers()
	{
		// Shadow map debug quad
		float AR = (float)height / (float)width;
		uboVSquad.projection = glm::ortho(2.5f / AR, 0.0f, 0.0f, 2.5f, -1.0f, 1.0f);
		uboVSquad.model = glm::mat4(1.0f);
		memcpy(uniformBuffers.debug.mapped, &uboVSquad, sizeof(uboVSquad));

		// 3D scene
		uboVSscene.projection = camera.matrices.perspective;
		uboVSscene.view = camera.matrices.view;
		uboVSscene.model = glm::mat4(1.0f);
		uboVSscene.lightPos = lightPos;
		uboVSscene.depthBiasMVP = uboOffscreenVS.depthMVP;
		memcpy(uniformBuffers.scene.mapped, &uboVSscene, sizeof(uboVSscene));
	}

	void updateUniformBufferOffscreen()
	{
		// Matrix from light's point of view
		glm::mat4 depthProjectionMatrix = glm::perspective(glm::radians(lightFOV), 1.0f, zNear, zFar);
		glm::mat4 depthViewMatrix = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0, 1, 0));
		glm::mat4 depthModelMatrix = glm::mat4(1.0f);

		uboOffscreenVS.depthMVP = depthProjectionMatrix * depthViewMatrix * depthModelMatrix;

		memcpy(uniformBuffers.offscreen.mapped, &uboOffscreenVS, sizeof(uboOffscreenVS));
	}

	// Find and create a compute capable device queue
	void getComputeQueue()
	{
		uint32_t queueFamilyCount;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);
		assert(queueFamilyCount >= 1);

		std::vector<VkQueueFamilyProperties> queueFamilyProperties;
		queueFamilyProperties.resize(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());

		// Some devices have dedicated compute queues, so we first try to find a queue that supports compute and not graphics
		bool computeQueueFound = false;
		for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++)
		{
			if ((queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && ((queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0))
			{
				compute.queueFamilyIndex = i;
				computeQueueFound = true;
				break;
			}
		}
		// If there is no dedicated compute queue, just find the first queue family that supports compute
		if (!computeQueueFound)
		{
			for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++)
			{
				if (queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
				{
					compute.queueFamilyIndex = i;
					computeQueueFound = true;
					break;
				}
			}
		}

		// Compute is mandatory in Vulkan, so there must be at least one queue family that supports compute
		assert(computeQueueFound);
		// Get a compute queue from the device
		vkGetDeviceQueue(device, compute.queueFamilyIndex, 0, &compute.queue);
	}

	// Prepare the compute pipeline that generates the ray traced image
	void prepareCompute()
	{
		getComputeQueue();

		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0: Storage image (raytraced output)
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				VK_SHADER_STAGE_COMPUTE_BIT,
				0),
			
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				VK_SHADER_STAGE_COMPUTE_BIT,
				1),
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				VK_SHADER_STAGE_COMPUTE_BIT,
				2),
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				VK_SHADER_STAGE_COMPUTE_BIT,
				3),
			
		};

		VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayout[0]));

		VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&compute.descriptorSetLayout[0],
				1);

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout[0]));

		VkDescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&compute.descriptorSetLayout[0],
				1);

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSet[0]));
		// Image descriptor for the shadow map attachment
		VkDescriptorImageInfo texDescriptor =
			vks::initializers::descriptorImageInfo(
				offscreenPass.depthSampler,
				offscreenPass.depth.view,
				VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);

		std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets =
		{
			// Binding 0: Output storage image
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet[0],
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				0,
				&VolumeInject.descriptor),
			// Binding 1: Uniform buffer block
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet[0],
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				1,
				&texDescriptor),
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet[0],
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				2,
				&compute.uniformBuffer.descriptor),
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet[0],
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				3,
				&uniformBuffers.scene.descriptor)
			
		};

		vkUpdateDescriptorSets(device, computeWriteDescriptorSets.size(), computeWriteDescriptorSets.data(), 0, NULL);

		// Create compute shader pipelines
		VkComputePipelineCreateInfo computePipelineCreateInfo =
			vks::initializers::computePipelineCreateInfo(
				compute.pipelineLayout[0],
				0);

		computePipelineCreateInfo.stage = loadShader(getAssetPath() + "shaders/shadowmapping/InjectLightingAndDensity.comp", VK_SHADER_STAGE_COMPUTE_BIT);
		VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipeline[0]));

		setLayoutBindings = {
			// Binding 0: Storage image (raytraced output)
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				VK_SHADER_STAGE_COMPUTE_BIT,
				0),
			// Binding 1: Uniform buffer block
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				VK_SHADER_STAGE_COMPUTE_BIT,
				1),
			

		};

		 descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayout[1]));

		pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&compute.descriptorSetLayout[1],
				1);

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout[1]));

	allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&compute.descriptorSetLayout[1],
				1);

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSet[1]));
		

		 computeWriteDescriptorSets =
		{
			// Binding 0: Output storage image
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet[1],
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				0,
				&VolumeScatter.descriptor),
			// Binding 1: Uniform buffer block
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet[1],
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				1,
				&VolumeInject.descriptor),
			

		};

		vkUpdateDescriptorSets(device, computeWriteDescriptorSets.size(), computeWriteDescriptorSets.data(), 0, NULL);

		// Create compute shader pipelines
		 computePipelineCreateInfo =
			vks::initializers::computePipelineCreateInfo(
				compute.pipelineLayout[1],
				0);

		computePipelineCreateInfo.stage = loadShader(getAssetPath() + "shaders/shadowmapping/Scatter.comp", VK_SHADER_STAGE_COMPUTE_BIT);
		VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipeline[1]));

	

		

		// Build a single command buffer containing the compute dispatch commands
		buildComputeCommandBuffer();
	}
	void buildComputeCommandBuffer()
	{ 	// Separate command pool as queue family for compute may be different than graphics
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = compute.queueFamilyIndex;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &compute.commandPool));

		// Create a command buffer for compute operations
		VkCommandBufferAllocateInfo cmdBufAllocateInfo =
			vks::initializers::commandBufferAllocateInfo(
				compute.commandPool,
				VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				1);

		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &compute.commandBuffer[0]));
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &compute.commandBuffer[1]));


		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &compute.semaphore[0]));
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &compute.semaphore[1]));

		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer[0], &cmdBufInfo));

		// Acquire barrier

		VkImageMemoryBarrier imageMemoryBarrier =
		{
			VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			nullptr,
			0,
			VK_ACCESS_SHADER_READ_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_UNDEFINED,
			graphics.queueFamilyIndex,
			compute.queueFamilyIndex,
			offscreenPass.depth.image,
			{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 }
		};

		vkCmdPipelineBarrier(
			compute.commandBuffer[0],
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageMemoryBarrier);
	



		vkCmdBindPipeline(compute.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline[0]);
		vkCmdBindDescriptorSets(compute.commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout[0], 0, 1, &compute.descriptorSet[0], 0, 0);
		vkCmdDispatch(compute.commandBuffer[0], VolumeInject.width / 16, VolumeInject.height / 16, VolumeInject.depth);
//---------------------------------------
		imageMemoryBarrier =
		{
			VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			nullptr,
			0,
			VK_ACCESS_SHADER_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_UNDEFINED,
			compute.queueFamilyIndex,
			graphics.queueFamilyIndex,
			offscreenPass.depth.image,
			{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 }
		};

		vkCmdPipelineBarrier(
			compute.commandBuffer[0],
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageMemoryBarrier);
//-------------------------------------------------------
		imageMemoryBarrier = {};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.image = VolumeInject.image;
		imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(
			compute.commandBuffer[0],
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_FLAGS_NONE,
			0, nullptr,
			0, nullptr,
			1, &imageMemoryBarrier);

		vkEndCommandBuffer(compute.commandBuffer[0]);
		//-------------------------------------------------

		VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer[1], &cmdBufInfo));
		vkCmdBindPipeline(compute.commandBuffer[1], VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline[1]);
		vkCmdBindDescriptorSets(compute.commandBuffer[1], VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout[1], 0, 1, &compute.descriptorSet[1], 0, 0);

		vkCmdDispatch(compute.commandBuffer[1], VolumeScatter.width / 16, VolumeScatter.height / 16, 1);
		
		//---------------------------------------------------------------
		imageMemoryBarrier = {};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.image = VolumeInject.image;
		imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(
			compute.commandBuffer[1],
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_FLAGS_NONE,
			0, nullptr,
			0, nullptr,
			1, &imageMemoryBarrier);
//---------------------------------------------------------------------------
	
		imageMemoryBarrier = {};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.image = VolumeScatter.image;
		imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		imageMemoryBarrier.srcQueueFamilyIndex = compute.queueFamilyIndex;
		imageMemoryBarrier.srcQueueFamilyIndex = graphics.queueFamilyIndex;

		vkCmdPipelineBarrier(
			compute.commandBuffer[1],
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_FLAGS_NONE,
			0, nullptr,
			0, nullptr,
			1, &imageMemoryBarrier);
//----------------------------------------------------------------------------
		vkEndCommandBuffer(compute.commandBuffer[1]);
		
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// Offscreen rendering

		// Wait for swap chain presentation to finish
		submitInfo.pWaitSemaphores = &semaphores.presentComplete;
		// Signal ready with offscreen semaphore
		submitInfo.pSignalSemaphores = &offscreenSemaphore;

		// Submit work
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &offScreenCmdBuffer;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		// Scene rendering
		// Wait for offscreen semaphore
		submitInfo.pWaitSemaphores = &offscreenSemaphore;
		// Signal ready with render complete semaphpre
		submitInfo.pSignalSemaphores = &compute.semaphore[0];
		submitInfo.pCommandBuffers = &compute.commandBuffer[0];

		VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &submitInfo, VK_NULL_HANDLE));

		// Scene rendering
		// Wait for offscreen semaphore
		submitInfo.pWaitSemaphores = &compute.semaphore[0];
		// Signal ready with render complete semaphpre
		submitInfo.pSignalSemaphores = &compute.semaphore[1];
		submitInfo.pCommandBuffers = &compute.commandBuffer[1];

		VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &submitInfo, VK_NULL_HANDLE));

		VK_CHECK_RESULT(vkQueueWaitIdle(compute.queue));
		
		submitInfo.pWaitSemaphores = &compute.semaphore[1];
		submitInfo.pSignalSemaphores = &semaphores.renderComplete;
		// Submit work
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		VulkanExampleBase::submitFrame();

	}

	void prepareGraphics()
	{
		graphics.queueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
		loadAssets();
		generateQuad();
		prepareOffscreenFramebuffer();
		prepareSceneRenderpass();
		prepareUniformBuffers();
		setupDescriptorSetLayout();
		
		setupDescriptorPool();
	
		
	}
	void prepare()
	{
		VulkanExampleBase::prepare();
		
		
		prepareGraphics();
		prepareTextureTarget(&VolumeInject, TEX_DIM, TEX_DIM,128, VK_FORMAT_R8G8B8A8_UNORM);
		prepareTextureTarget(&VolumeScatter, TEX_DIM, TEX_DIM, 128, VK_FORMAT_R8G8B8A8_UNORM);
		setupDescriptorSets();
		preparePipelines();
		
		buildDeferredCommandBuffer();
		prepareCompute();
		
		buildCommandBuffers();
		
		
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
		if (!paused || camera.updated)
		{
			updateLight();
			updateUniformBufferOffscreen();
			updateUniformBuffers();
			updateComputeUniformBuffer();
		}
	}

	virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
	{
		if (overlay->header("Settings")) {
			if (overlay->comboBox("Scenes", &sceneIndex, sceneNames)) {
				buildCommandBuffers();
			}
			if (overlay->checkBox("Display shadow render target", &displayShadowMap)) {
				buildCommandBuffers();
			}
			if (overlay->checkBox("PCF filtering", &filterPCF)) {
				buildCommandBuffers();
			}
		}
	}
};

std::shared_ptr<VulkanExample> vulkanExample;                                                                   \
LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)						\
{																									\
if (vulkanExample != NULL)																		\
{																								\
vulkanExample->handleMessages(hWnd, uMsg, wParam, lParam);									\
}																								\
return (DefWindowProc(hWnd, uMsg, wParam, lParam));												\
}																									\
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int)									\
{																									\
for (int32_t i = 0; i < __argc; i++) { VulkanExample::args.push_back(__argv[i]); };  			\
	vulkanExample.reset(new VulkanExample());															\
	vulkanExample->initVulkan();																	\
	vulkanExample->setupWindow(hInstance, WndProc);													\
	vulkanExample->prepare();																		\
	vulkanExample->renderLoop();																	\
	
	return 0;																						\
}
