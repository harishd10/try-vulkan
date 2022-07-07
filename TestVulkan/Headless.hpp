#ifndef HEADLESS_HPP
#define HEADLESS_HPP


#include <vulkan/vulkan.hpp>

class Headless
{
public:
    Headless();
    ~Headless();

public:
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    uint32_t queueFamilyIndex;
    vk::UniqueDevice device;
    vk::Queue queue;

    vk::UniqueCommandPool commandPool;
    vk::UniqueCommandBuffer commandBuffer;

    bool useDepth;
    vk::Format colorFormat, depthFormat;
    struct FrameBufferAttachment {
        vk::UniqueImage image;
        vk::UniqueDeviceMemory memory;
        vk::UniqueImageView view;
    };
    int32_t width, height;
    FrameBufferAttachment colorAttachment, depthAttachment;

    vk::UniqueRenderPass renderPass;
    vk::UniqueFramebuffer framebuffer;

    struct Vertex {
        float position[2];
    };
    vk::UniqueShaderModule vertexShaderModule, fragmentShaderModule;
    vk::UniqueBuffer vertexBuffer;
    vk::UniqueDeviceMemory vertexMemory, indexMemory;

    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipelineCache pipelineCache;
    vk::UniquePipeline pipeline;

    /////////// Still unused
    vk::DebugUtilsMessengerEXT debugMessenger;
    int vsize;

protected:
    bool createInstance();
    bool createDevice(vk::PhysicalDeviceType type);
    bool createCommandPool();
    void createCommandBuffer(vk::UniqueCommandBuffer &commandBuffer);

    bool createFrameBufferAttachments(int32_t width, int32_t height,
                                      vk::Format colorFormat = vk::Format::eR32G32B32A32Sfloat,
                                      bool useDepth = true, vk::Format depthFormat = vk::Format::eD16Unorm);


    void createRenderPass();
    void createFrameBuffer();
    void setupShaders();

    void initData();

    void setupGraphicsPipeline();
    void finallyDraw();

    void saveImageToFile();
    /////////////// TODO
    /* Copy framebuffer image to host visible image */
    /* Save host visible framebuffer image to disk (ppm format) */

};

#endif // HEADLESS_HPP
