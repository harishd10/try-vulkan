#include "Headless.hpp"

#include "utils.h"

#include <vector>
#include <iostream>
#include <fstream>

#include <QDebug>
#include <QFile>
#include <QDataStream>
#include <QImage>
#include <QMatrix4x4>
#include <QElapsedTimer>

#ifdef DEV_BUILD
inline static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
    if(messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    }
    return VK_FALSE;
}
#endif

Headless::Headless() {
    try {
        validate(createInstance(),"createInstance");
        validate(createDevice(vk::PhysicalDeviceType::eDiscreteGpu),"createDevice");
        validate(createCommandPool(),"createCommandBuffer");
        validate(createFrameBufferAttachments(1024,1024,vk::Format::eR32G32B32A32Sfloat),"createFrameBufferAttachments");
        createRenderPass();
        createFrameBuffer();
        setupShaders();
        initData();
        setupGraphicsPipeline();

        qDebug() << "finally drawing after all the setup...";
        QElapsedTimer timer;
        timer.start();
        finallyDraw();
        qint64 t = timer.elapsed();
        qDebug() << "rendering time: " << t;

        saveImageToFile();
    } catch (vk::SystemError err) {
        qDebug() << "ERROR: system error - " << QString(err.what());
        exit(-1);
    } catch (std::runtime_error err) {
        qDebug() << "ERROR: runtime error - " << QString(err.what());
        exit(-1);
    } catch(...) {
        qDebug() << "ERROR: unknown error";
        exit(-1);
    }

}

bool Headless::createInstance() {
    std::vector<const char*> layers;
    std::vector<const char*> extensions;
#ifdef DEV_BUILD
    std::vector<const char*> validationLayers = {
        "VK_LAYER_LUNARG_standard_validation",
        "VK_LAYER_KHRONOS_validation"
    };

    auto installedLayers = vk::enumerateInstanceLayerProperties();

    for (auto &w : validationLayers) {
        for (auto &i : installedLayers) {
            if (std::string(i.layerName.data()).compare(w) == 0) {
                layers.emplace_back(w);
                break;
            }
        }
    }
    const char *validationExt = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    extensions.push_back(validationExt);
#endif

    vk::ApplicationInfo appInfo("App Name",1,"engineName",1,VK_MAKE_VERSION(1, 3, 0));
    vk::InstanceCreateInfo instanceInfo({},&appInfo,layers.size(),layers.data(),extensions.size(),extensions.data());
    vk::Result res = vk::createInstance(&instanceInfo, nullptr, &this->instance);
    if(res != vk::Result::eSuccess) {
        qDebug() << "*********************** could not create instance ***********************";
        return false;
    }

#ifdef DEV_BUILD
    if(layers.size() > 0) {
        vk::DebugUtilsMessengerCreateInfoEXT debugInfo;
        debugInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
        debugInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
        debugInfo.pfnUserCallback = debugCallback;

        auto func = (PFN_vkCreateDebugUtilsMessengerEXT) instance.getProcAddr("vkCreateDebugUtilsMessengerEXT");
        if (func != nullptr) {
            VkResult res = func(instance, reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(&debugInfo), nullptr, reinterpret_cast<VkDebugUtilsMessengerEXT*>(&debugMessenger));
            if(res != VK_SUCCESS) {
                qDebug() << "*********************** could not setup debug messenger ***********************";
                return false;
            }
        } else {
            qDebug() << "ERROR: could not find debug function!";
            return false;
        }
    }
#endif
    return true;
}

bool Headless::createDevice(vk::PhysicalDeviceType type) {
    std::vector<vk::PhysicalDevice> physicalDevices = this->instance.enumeratePhysicalDevices();
    qDebug() << "no. of devices: " << physicalDevices.size();
    if(physicalDevices.size() == 0) {
        qDebug() << "ERROR: found no vulkan devices!";
        return false;
    }
    int devId = -1;
    vk::PhysicalDeviceProperties props;
    for(int i = 0;i < physicalDevices.size();i ++) {
        props = physicalDevices[i].getProperties();
        if(props.deviceType == type) {
            devId = i;
            break;
        }
    }
    if(devId == -1) {
        qDebug() << "ERROR: could not find vulkan device of type" << (int) type << "!";
        return false;
    }
    this->physicalDevice = physicalDevices[devId];
    qDebug() << "Using device:" << QString(props.deviceName);

    // get the QueueFamilyProperties of the PhysicalDevice
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    // get the first index into queueFamiliyProperties which supports graphics
    qDebug() << "No. of queues: " << queueFamilyProperties.size();
    queueFamilyIndex = std::distance(queueFamilyProperties.begin(),
                                                    std::find_if(queueFamilyProperties.begin(),
                                                                 queueFamilyProperties.end(),
                                                                 [](vk::QueueFamilyProperties const& qfp) { return qfp.queueFlags & vk::QueueFlagBits::eGraphics; }));
    if(queueFamilyIndex < 0 || queueFamilyIndex >= queueFamilyProperties.size()) {
        qDebug() << "ERROR: could not find graphics queue!";
        return false;
    }
    qDebug() << "graphics queue found at" << queueFamilyIndex;

    // create a logical device. using UniqueDevice so it gets destroyed automatically
    float queuePriority = 0.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), static_cast<uint32_t>(queueFamilyIndex), 1, &queuePriority);
    device = physicalDevice.createDeviceUnique(vk::DeviceCreateInfo(vk::DeviceCreateFlags(), 1, &deviceQueueCreateInfo));
    qDebug() << "successfully created logical device";

    queue = device->getQueue(queueFamilyIndex,0);
    qDebug() << "successfully obtained graphics queue";

    return true;
}

void Headless::createCommandBuffer(vk::UniqueCommandBuffer &commandBuffer) {
    // allocate a CommandBuffer from the CommandPool
    std::vector<vk::UniqueCommandBuffer> commandBuffers = device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(commandPool.get(), vk::CommandBufferLevel::ePrimary, 1));
    if(commandBuffers.size() == 0) {
        qDebug() << "ERROR: could not create command buffer!";
        exit(-2);
    }
    commandBuffer = std::move(commandBuffers[0]);
    qDebug() << "created command buffer";
}

bool Headless::createCommandPool() {
    // create a UniqueCommandPool to allocate a CommandBuffer from
    this->commandPool = device->createCommandPoolUnique(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex));
    qDebug() << "created command pool";

    return true;
}

/**
 * This is for offscreen rendering, where there is not X server.
 * IF using X server, then have to create swapchain for the device etc etc...
 */
bool Headless::createFrameBufferAttachments(int32_t width, int32_t height, vk::Format colorFormat, bool useDepth, vk::Format depthFormat) {
    this->width = width;
    this->height = height;
    this->colorFormat = colorFormat;
    this->depthFormat = depthFormat;
    this->useDepth = useDepth;
    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
    vk::ComponentMapping componentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA);

    {
        vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(colorFormat);
        vk::ImageTiling tiling;
        if (formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eColorAttachment) {
            tiling = vk::ImageTiling::eOptimal;
        } else if (formatProperties.linearTilingFeatures & vk::FormatFeatureFlagBits::eColorAttachment) {
            tiling = vk::ImageTiling::eLinear;
        } else {
            throw std::runtime_error("ColorAttachment is not supported for given color format.");
        }
        vk::ImageCreateInfo imageCreateInfo(vk::ImageCreateFlags(),
                                            vk::ImageType::e2D,
                                            colorFormat,
                                            vk::Extent3D(width,height,1),
                                            1, 1,
                                            vk::SampleCountFlagBits::e1,
                                            tiling,
                                            vk::ImageUsageFlagBits::eColorAttachment|vk::ImageUsageFlagBits::eTransferSrc);
        colorAttachment.image = device->createImageUnique(imageCreateInfo);


        // TODO Redundant code... create inline function
        vk::MemoryRequirements memoryRequirements = device->getImageMemoryRequirements(colorAttachment.image.get());
        uint32_t typeBits = memoryRequirements.memoryTypeBits;
        uint32_t typeIndex = uint32_t(~0);
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
            if ((typeBits & 1) && ((memoryProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) == vk::MemoryPropertyFlagBits::eDeviceLocal)) {
                typeIndex = i;
                break;
            }
            typeBits >>= 1;
        }
        if(typeIndex == ~0) {
            qDebug() << "ERROR: depth type bit error";
            return false;
        }
        colorAttachment.memory = device->allocateMemoryUnique(vk::MemoryAllocateInfo(memoryRequirements.size, typeIndex));
        device->bindImageMemory(colorAttachment.image.get(), colorAttachment.memory.get(), 0);

        vk::ImageSubresourceRange subResourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        colorAttachment.view = device->createImageViewUnique(vk::ImageViewCreateInfo(vk::ImageViewCreateFlags(), colorAttachment.image.get(), vk::ImageViewType::e2D, colorFormat, componentMapping, subResourceRange));

        qDebug() << "created color attachment";
    }

    // Depth attachment
    // TODO Assuming no stencil
    if(useDepth) {
        vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(depthFormat);
        vk::ImageTiling tiling;
        if (formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
            tiling = vk::ImageTiling::eOptimal;
        } else if (formatProperties.linearTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
            tiling = vk::ImageTiling::eLinear;
        } else {
            throw std::runtime_error("DepthStencilAttachment is not supported for given depth format.");
        }
        vk::ImageCreateInfo imageCreateInfo(vk::ImageCreateFlags(),
                                            vk::ImageType::e2D,
                                            depthFormat,
                                            vk::Extent3D(width,height,1),
                                            1, 1,
                                            vk::SampleCountFlagBits::e1,
                                            tiling,
                                            vk::ImageUsageFlagBits::eDepthStencilAttachment);
        depthAttachment.image = device->createImageUnique(imageCreateInfo);

        vk::MemoryRequirements memoryRequirements = device->getImageMemoryRequirements(depthAttachment.image.get());
        uint32_t typeBits = memoryRequirements.memoryTypeBits;
        uint32_t typeIndex = uint32_t(~0);
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
            if ((typeBits & 1) && ((memoryProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) == vk::MemoryPropertyFlagBits::eDeviceLocal)) {
                typeIndex = i;
                break;
            }
            typeBits >>= 1;
        }
        if(typeIndex == ~0) {
            qDebug() << "ERROR: depth type bit error";
            return false;
        }
        depthAttachment.memory = device->allocateMemoryUnique(vk::MemoryAllocateInfo(memoryRequirements.size, typeIndex));
        device->bindImageMemory(depthAttachment.image.get(), depthAttachment.memory.get(), 0);

        // TODO If stencil present, then need to update subResourceRange
        vk::ImageSubresourceRange subResourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);
        depthAttachment.view = device->createImageViewUnique(vk::ImageViewCreateInfo(vk::ImageViewCreateFlags(), depthAttachment.image.get(), vk::ImageViewType::e2D, depthFormat, componentMapping, subResourceRange));
        qDebug() << "created depth buffer attachment";
    }
    return true;
}

void Headless::createRenderPass() {
    vk::AttachmentDescription attachmentDescriptions[2];
    attachmentDescriptions[0] = vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), colorFormat, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
                                                          vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal);
    attachmentDescriptions[1] = vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), depthFormat, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
                                                          vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vk::AttachmentReference colorReference(0, vk::ImageLayout::eColorAttachmentOptimal);
    vk::AttachmentReference depthReference(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    vk::SubpassDescription subpass(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &colorReference, nullptr, &depthReference);

    // TODO check what happens without the dependencies
    vk::SubpassDependency subpassDependencies[2];
    subpassDependencies[0] = vk::SubpassDependency(VK_SUBPASS_EXTERNAL,0,vk::PipelineStageFlagBits::eBottomOfPipe,vk::PipelineStageFlagBits::eColorAttachmentOutput,vk::AccessFlagBits::eMemoryRead,vk::AccessFlagBits::eColorAttachmentRead|vk::AccessFlagBits::eColorAttachmentWrite,vk::DependencyFlagBits::eByRegion);
    subpassDependencies[1] = vk::SubpassDependency(0,VK_SUBPASS_EXTERNAL,vk::PipelineStageFlagBits::eColorAttachmentOutput,vk::PipelineStageFlagBits::eBottomOfPipe,vk::AccessFlagBits::eColorAttachmentRead|vk::AccessFlagBits::eColorAttachmentWrite,vk::AccessFlagBits::eMemoryRead,vk::DependencyFlagBits::eByRegion);

    renderPass = device->createRenderPassUnique(vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), 2, attachmentDescriptions, 1, &subpass, 2, subpassDependencies));

    qDebug() << "created render pass";
}

void Headless::createFrameBuffer() {
    vk::ImageView attachments[2];
    attachments[0] = colorAttachment.view.get();
    attachments[1] = depthAttachment.view.get();
    framebuffer = device->createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), renderPass.get(), 2, attachments, this->width, this->height, 1));
    qDebug() << "created framebuffer";
}

void Headless::setupShaders() {
    std::vector<uint32_t> vshader, fshader;
    validate(readShader(":/shaders/triangle.vert.spv",vshader),"read vertex shader");
    validate(readShader(":/shaders/triangle.frag.spv",fshader),"read fragment shader");

    vk::ShaderModuleCreateInfo vertexShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), vshader.size() * sizeof(uint32_t), vshader.data());
    vertexShaderModule = device->createShaderModuleUnique(vertexShaderModuleCreateInfo);

    vk::ShaderModuleCreateInfo fragmentShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), fshader.size() * sizeof(uint32_t), fshader.data());
    fragmentShaderModule = device->createShaderModuleUnique(fragmentShaderModuleCreateInfo);

    qDebug() << "setup shader";
}


inline uint32_t findMemoryType(vk::PhysicalDeviceMemoryProperties const& memoryProperties, uint32_t typeBits, vk::MemoryPropertyFlags requirementsMask) {
    uint32_t typeIndex = uint32_t(~0);
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if ((typeBits & 1) && ((memoryProperties.memoryTypes[i].propertyFlags & requirementsMask) == requirementsMask)) {
            typeIndex = i;
            break;
        }
        typeBits >>= 1;
    }
    if(typeIndex == ~0) {
        qDebug() << "ERROR: findMemoryType";
        exit(-2);
    }
    return typeIndex;
}

//TODO Staging buffer not used. Need to figure it out
void Headless::initData() {
    std::vector<Vertex> vertices;
    vsize = 100000;
    for(int i = 0;i < vsize;i ++) {
        float x = float(rand()) / RAND_MAX;
        float y = float(rand()) / RAND_MAX;

        x = (x - 0.5) * 2;
        y = (y - 0.5) * 2;
        Vertex v = {x,y};
        vertices.push_back(v);
    }

    {
        // create a vertex buffer for some vertex and color data
        vertexBuffer = device->createBufferUnique(vk::BufferCreateInfo(
                                                      vk::BufferCreateFlags(),
                                                      vertices.size() * sizeof(Vertex),
                                                      vk::BufferUsageFlagBits::eVertexBuffer));
        // allocate device memory for that buffer
        vk::MemoryRequirements memoryRequirements = device->getBufferMemoryRequirements(vertexBuffer.get());
        uint32_t memoryTypeIndex = findMemoryType(physicalDevice.getMemoryProperties(), memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        vertexMemory = device->allocateMemoryUnique(vk::MemoryAllocateInfo(memoryRequirements.size, memoryTypeIndex));

        // copy the vertex and color data into that device memory
        uint8_t *pData = static_cast<uint8_t*>(device->mapMemory(vertexMemory.get(), 0, memoryRequirements.size));
        memcpy(pData, vertices.data(), vertices.size() * sizeof(Vertex));
        device->unmapMemory(vertexMemory.get());

        // and bind the device memory to the vertex buffer
        device->bindBufferMemory(vertexBuffer.get(), vertexMemory.get(), 0);
    }

    qDebug() << "setup data in vertexx buffer";
}

void Headless::setupGraphicsPipeline() {
    pipelineLayout = device->createPipelineLayoutUnique(
                vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(),
                                             0, nullptr,                // descriptorSetLayout
                                             0, nullptr                 // constantRange
                                             ));

    pipelineCache = device->createPipelineCacheUnique(
                vk::PipelineCacheCreateInfo(vk::PipelineCacheCreateFlags())
                );


    // Create Pipeline
    vk::PipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::ePointList);

    vk::PipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo
    (
        vk::PipelineRasterizationStateCreateFlags(),  // flags
        false,                                        // depthClampEnable
        false,                                        // rasterizerDiscardEnable
        vk::PolygonMode::eFill,                       // polygonMode
        vk::CullModeFlagBits::eBack,                  // cullMode
        vk::FrontFace::eClockwise,                    // frontFace
        false,                                        // depthBiasEnable
        0.0f,                                         // depthBiasConstantFactor
        0.0f,                                         // depthBiasClamp
        0.0f,                                         // depthBiasSlopeFactor
        1.0f                                          // lineWidth
    );

    vk::ColorComponentFlags colorComponentFlags(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    vk::PipelineColorBlendAttachmentState pipelineColorBlendAttachmentState
    (
        false,                      // blendEnable
        vk::BlendFactor::eZero,     // srcColorBlendFactor
        vk::BlendFactor::eZero,     // dstColorBlendFactor
        vk::BlendOp::eAdd,          // colorBlendOp
        vk::BlendFactor::eZero,     // srcAlphaBlendFactor
        vk::BlendFactor::eZero,     // dstAlphaBlendFactor
        vk::BlendOp::eAdd,          // alphaBlendOp
        colorComponentFlags         // colorWriteMask
    );

    vk::PipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo
    (
        vk::PipelineColorBlendStateCreateFlags(),   // flags
        false,                                      // logicOpEnable
        vk::LogicOp::eNoOp,                         // logicOp
        1,                                          // attachmentCount
        &pipelineColorBlendAttachmentState,         // pAttachments
        { { (1.0f, 1.0f, 1.0f, 1.0f) } }            // blendConstants
    );

    vk::StencilOpState stencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eAlways);
    vk::PipelineDepthStencilStateCreateInfo pipelineDepthStencilStateCreateInfo
    (
        vk::PipelineDepthStencilStateCreateFlags(), // flags
        true,                                       // depthTestEnable
        true,                                       // depthWriteEnable
        vk::CompareOp::eLessOrEqual,                // depthCompareOp
        false,                                      // depthBoundTestEnable
        false,                                      // stencilTestEnable
        stencilOpState,                             // front
        stencilOpState                              // back
    );

    vk::PipelineViewportStateCreateInfo pipelineViewportStateCreateInfo(vk::PipelineViewportStateCreateFlags(), 1, nullptr, 1, nullptr);

    vk::PipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo;

    vk::DynamicState dynamicStates[2] = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
    vk::PipelineDynamicStateCreateInfo pipelineDynamicStateCreateInfo(vk::PipelineDynamicStateCreateFlags(), 2, dynamicStates);

    vk::PipelineShaderStageCreateInfo pipelineShaderStageCreateInfos[2] =
    {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex, vertexShaderModule.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment, fragmentShaderModule.get(), "main")
    };

    vk::VertexInputBindingDescription vertexInputBindingDescription(0, sizeof(Vertex));
    vk::VertexInputAttributeDescription vertexInputAttributeDescriptions[1] =
    {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, 0),                  // coordinates
    };
    vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo(
        vk::PipelineVertexInputStateCreateFlags(),  // flags
        1,                                          // vertexBindingDescriptionCount
        &vertexInputBindingDescription,             // pVertexBindingDescription
        1,                                          // vertexAttributeDescriptionCount
        vertexInputAttributeDescriptions            // pVertexAttributeDescriptions
    );

    vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo
    (
        vk::PipelineCreateFlags(),                  // flags
        2,                                          // stageCount
        pipelineShaderStageCreateInfos,             // pStages
        &pipelineVertexInputStateCreateInfo,        // pVertexInputState
        &pipelineInputAssemblyStateCreateInfo,      // pInputAssemblyState
        nullptr,                                    // pTessellationState
        &pipelineViewportStateCreateInfo,           // pViewportState
        &pipelineRasterizationStateCreateInfo,      // pRasterizationState
        &pipelineMultisampleStateCreateInfo,        // pMultisampleState
        &pipelineDepthStencilStateCreateInfo,       // pDepthStencilState
        &pipelineColorBlendStateCreateInfo,         // pColorBlendState
        &pipelineDynamicStateCreateInfo,            // pDynamicState
        pipelineLayout.get(),                       // layout
        renderPass.get()                            // renderPass
    );

    pipeline = device->createGraphicsPipelineUnique(pipelineCache.get(), graphicsPipelineCreateInfo);
    qDebug() << "created graphics pipeline";
}

void Headless::finallyDraw() {
    this->createCommandBuffer(commandBuffer);
    commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));

    vk::ClearValue clearValues[2];
    clearValues[0].color = vk::ClearColorValue(std::array<float, 4>({ 0.0f, 0.0f, 0.0f, 1.0f }));
    clearValues[1].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

    // TODO  use surfaceData similar to Hpp example
    vk::RenderPassBeginInfo renderPassBeginInfo(renderPass.get(), framebuffer.get(), vk::Rect2D(vk::Offset2D(0, 0), vk::Extent2D(width,height)), 2, clearValues);

    commandBuffer->beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

    vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
    commandBuffer->setViewport(0, viewport);

    vk::Rect2D scissor(vk::Offset2D(0, 0), vk::Extent2D(width,height));
    commandBuffer->setScissor(0, scissor);

    commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.get());

    vk::DeviceSize offset = 0;
    commandBuffer->bindVertexBuffers(0, vertexBuffer.get(), offset);

    commandBuffer->draw(vsize,vsize,0,0);
    commandBuffer->endRenderPass();
    commandBuffer->end();

    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer.get());
    vk::UniqueFence drawFence = device->createFenceUnique(vk::FenceCreateInfo());
    queue.submit(submitInfo,drawFence.get());
    device->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);
    device->waitIdle();

    qDebug() << "finished drawing!!!!!";
}

inline QImage dataToQImage( int width, int height, const float *data) {
    QImage image( width, height, QImage::Format_ARGB32 );
    for ( int i = 0; i < width * height; ++i )
    {
        int index = i * 4;
        QRgb argb = qRgba( data[index + 0] * 255, //red
                           data[index + 1] * 255, //green
                           data[index + 2] * 255, //blue
                           data[index + 3] * 255);   //alpha
        image.setPixel(i % width, i / width, argb );
    }
    return image;
}

void Headless::saveImageToFile() {
    // Create the image
    vk::UniqueImage dstImage = device->createImageUnique(
                vk::ImageCreateInfo(vk::ImageCreateFlags(),vk::ImageType::e2D,
                                    vk::Format::eR32G32B32A32Sfloat,vk::Extent3D(width,height,1),
                                    1,1,vk::SampleCountFlagBits::e1,vk::ImageTiling::eLinear,
                                    vk::ImageUsageFlagBits::eTransferDst));

    // Create memory to back up the image
    vk::MemoryRequirements memRequirements = device->getImageMemoryRequirements(dstImage.get());
    // Memory must be host visible to copy from
    uint32_t typeIndex = findMemoryType(physicalDevice.getMemoryProperties(),memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::UniqueDeviceMemory dstImageMemory = device->allocateMemoryUnique(vk::MemoryAllocateInfo(memRequirements.size, typeIndex));
    device->bindImageMemory(dstImage.get(), dstImageMemory.get(), 0);

    // Do the actual blit from the offscreen image to our host visible destination image
    commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));

    // Transition destination image to transfer destination layout
    vk::ImageMemoryBarrier barrier(vk::AccessFlags(),vk::AccessFlagBits::eTransferWrite,
                                   vk::ImageLayout::eUndefined,vk::ImageLayout::eTransferDstOptimal,
                                   VK_QUEUE_FAMILY_IGNORED,VK_QUEUE_FAMILY_IGNORED,
                                   dstImage.get(),vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor,0,1,0,1));
    commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags(),
                                   nullptr,nullptr,
                                   barrier);

    // colorAttachment.image is already in VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, and does not need to be transitioned
    vk::ImageCopy imgCpy(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor,0,0,1),
                         vk::Offset3D(),
                         vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor,0,0,1),
                         vk::Offset3D(),vk::Extent3D(width,height,1));
    commandBuffer->copyImage(colorAttachment.image.get(), vk::ImageLayout::eTransferSrcOptimal,
                             dstImage.get(),vk::ImageLayout::eTransferDstOptimal,
                             imgCpy);


    // Transition destination image to general layout, which is the required layout for mapping the image memory later on
    vk::ImageMemoryBarrier barrier2(vk::AccessFlagBits::eTransferWrite,vk::AccessFlagBits::eMemoryRead,
                                   vk::ImageLayout::eTransferDstOptimal,vk::ImageLayout::eGeneral,
                                   VK_QUEUE_FAMILY_IGNORED,VK_QUEUE_FAMILY_IGNORED,
                                   dstImage.get(),vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor,0,1,0,1));
    commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags(),
                                   nullptr,nullptr,
                                   barrier2);

    commandBuffer->end();
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer.get());
    vk::UniqueFence drawFence = device->createFenceUnique(vk::FenceCreateInfo());
    queue.submit(submitInfo,drawFence.get());
    device->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);

    // Get layout of the image (including row pitch)
    vk::SubresourceLayout subResourceLayout = device->getImageSubresourceLayout(dstImage.get(),vk::ImageSubresource(vk::ImageAspectFlagBits::eColor));
    // Map image memory so we can start copying from it
    const float* imagedata = (float*)device->mapMemory(dstImageMemory.get(),0,VK_WHOLE_SIZE);
    imagedata += subResourceLayout.offset;

    qDebug() << "mapped image to host";

    QImage qimg = dataToQImage(width,height,imagedata);
    qimg.save("../test.jpg");
    qDebug() << "saved image to file";
}

Headless::~Headless() {
#ifdef DEV_BUILD
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) instance.getProcAddr("vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger,nullptr);
    } else {
        qDebug() << "ERROR: could not find debug destroy function!";
    }
#endif
    qDebug() << "successfully destroyed instance";
}
