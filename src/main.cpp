#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <fstream>
#include <glm/glm.hpp>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>

std::vector<uint8_t> readFile(const char* path) {
  std::ifstream file;
  // Since there are actually a few places where we might run the executable
  // from (e.g. from the build directory, or from the root directory), we have
  // to try those places too.
  static const char* prefixes[] = {"", "../", "build/"};
  for (const char* prefix : prefixes) {
    file.open(std::string(prefix) + path, std::ios::ate | std::ios::binary);
    if (file.is_open()) {
      break;
    }
  }
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file");
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<uint8_t> buffer(fileSize);

  file.seekg(0);
  file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

  return buffer;
}

std::vector<const char*> getValidationLayers() {
#ifndef NDEBUG
  return {"VK_LAYER_KHRONOS_validation"};
#else
  return {};
#endif
}

GLFWwindow* createGLFWWindow() {
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
  const GLFWvidmode* videoMode = glfwGetVideoMode(primaryMonitor);

  return glfwCreateWindow(videoMode->width, videoMode->height, "Vulkan test",
                          primaryMonitor, nullptr);
}

vk::Instance createVulkanInstance() {
  vk::ApplicationInfo appInfo = {
      .pApplicationName = "Vulkan test",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_0,
  };

  uint32_t glfwExtensionCount = 0;
  const char** glfwExtensions =
      glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  auto validationLayers = getValidationLayers();

  vk::InstanceCreateInfo createInfo = {
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = static_cast<uint32_t>(validationLayers.size()),
      .ppEnabledLayerNames = validationLayers.data(),
      .enabledExtensionCount = glfwExtensionCount,
      .ppEnabledExtensionNames = glfwExtensions,
  };

  return vk::createInstance(createInfo);
}

vk::SurfaceKHR createSurface(const vk::Instance& instance, GLFWwindow* window) {
  VkSurfaceKHR surface;
  if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create window surface");
  }
  return surface;
}

struct RequiredQueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;

  RequiredQueueFamilyIndices(const vk::PhysicalDevice& physicalDevice,
                             const vk::SurfaceKHR& surface) {
    auto queueFamilies = physicalDevice.getQueueFamilyProperties();

    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
      const auto& queueFamily = queueFamilies[i];

      if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
        graphicsFamily = i;
      } else if (physicalDevice.getSurfaceSupportKHR(i, surface)) {
        presentFamily = i;
      }

      if (isComplete()) {
        break;
      }
    }
  }

  bool isComplete() const {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

static std::array<const char*, 1> requiredExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

bool requiredExtensionsSupported(const vk::PhysicalDevice& physicalDevice) {
  auto availableExtensions =
      physicalDevice.enumerateDeviceExtensionProperties();

  for (const auto& requiredExtension : requiredExtensions) {
    bool found = false;
    for (const auto& availableExtension : availableExtensions) {
      if (strcmp(requiredExtension, availableExtension.extensionName) == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

struct SwapChainCapabilities {
  vk::SurfaceCapabilitiesKHR capabilities;
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> presentModes;

  SwapChainCapabilities(const vk::PhysicalDevice& physicalDevice,
                        const vk::SurfaceKHR& surface) {
    capabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
    formats = physicalDevice.getSurfaceFormatsKHR(surface);
    presentModes = physicalDevice.getSurfacePresentModesKHR(surface);
  }

  bool isAdequate() const { return !formats.empty() && !presentModes.empty(); }

  vk::SurfaceFormatKHR chooseFormat() const {
    // We ideally go with SRGB BGRA, but otherwise fall back to the first one in
    // the list.
    for (const auto& format : formats) {
      if (format.format == vk::Format::eB8G8R8A8Srgb &&
          format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
        return format;
      }
    }
    return formats[0];
  }
  vk::PresentModeKHR choosePresentMode() const {
    // We ideally go with triple buffering, but otherwise fall back to the first
    // one in the list.
    for (const auto& presentMode : presentModes) {
      if (presentMode == vk::PresentModeKHR::eMailbox) {
        return presentMode;
      }
    }
    return vk::PresentModeKHR::eFifo;
  }
  vk::Extent2D chooseExtent(GLFWwindow* window) const {
    // If the current extent is something specific, we go with that. Otherwise,
    // we use the size of the window.
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    vk::Extent2D actualExtent = {static_cast<uint32_t>(width),
                                 static_cast<uint32_t>(height)};

    actualExtent.width =
        std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.width);
    actualExtent.height =
        std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                   capabilities.maxImageExtent.height);

    return actualExtent;
  }
};

vk::PhysicalDevice pickPhysicalDevice(const vk::Instance& instance,
                                      const vk::SurfaceKHR& surface) {
  auto physicalDevices = instance.enumeratePhysicalDevices();

  std::optional<vk::PhysicalDevice> selectedPhysicalDevice;

  for (const auto& physicalDevice : physicalDevices) {
    RequiredQueueFamilyIndices queueFamilyIndices(physicalDevice, surface);

    if (queueFamilyIndices.isComplete() &&
        requiredExtensionsSupported(physicalDevice) &&
        SwapChainCapabilities(physicalDevice, surface).isAdequate()) {
      selectedPhysicalDevice = physicalDevice;
      break;
    }
  }

  if (!selectedPhysicalDevice) {
    throw std::runtime_error("Failed to find a suitable GPU");
  }
  return *selectedPhysicalDevice;
}

vk::Device createLogicalDevice(const vk::PhysicalDevice& physicalDevice,
                               const RequiredQueueFamilyIndices& queueIndices) {
  float queuePriority = 1.0f;
  std::set<uint32_t> uniqueQueueFamilies = {*queueIndices.graphicsFamily,
                                            *queueIndices.presentFamily};

  std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
  for (uint32_t queueFamily : uniqueQueueFamilies) {
    vk::DeviceQueueCreateInfo queueCreateInfo = {
        .queueFamilyIndex = queueFamily,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };
    queueCreateInfos.push_back(queueCreateInfo);
  }

  vk::DeviceCreateInfo createInfo = {
      .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
      .pQueueCreateInfos = queueCreateInfos.data(),
      .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
      .ppEnabledExtensionNames = requiredExtensions.data(),
  };

  return physicalDevice.createDevice(createInfo);
}

struct Queues {
  vk::Queue graphicsQueue;
  vk::Queue presentQueue;

  Queues(const vk::Device& device,
         const RequiredQueueFamilyIndices& queueIndices) {
    graphicsQueue = device.getQueue(*queueIndices.graphicsFamily, 0);
    presentQueue = device.getQueue(*queueIndices.presentFamily, 0);
  }
};

vk::SwapchainKHR createSwapChain(const vk::Device& device,
                                 const vk::SurfaceKHR& surface,
                                 const SwapChainCapabilities& capabilities,
                                 const RequiredQueueFamilyIndices& queueIndices,
                                 const Queues& queues,
                                 const vk::Extent2D& extent) {
  uint32_t imageCount = capabilities.capabilities.minImageCount + 1;
  if (capabilities.capabilities.maxImageCount > 0 &&
      imageCount > capabilities.capabilities.maxImageCount) {
    imageCount = capabilities.capabilities.maxImageCount;
  }

  vk::SharingMode sharingMode = vk::SharingMode::eExclusive;
  uint32_t queueFamilyIndexCount = 0;  // (0 = no sharing)
  if (*queueIndices.graphicsFamily != *queueIndices.presentFamily) {
    sharingMode = vk::SharingMode::eConcurrent;
    queueFamilyIndexCount = 2;
  }

  uint32_t queueFamilyIndices[2] = {*queueIndices.graphicsFamily,
                                    *queueIndices.presentFamily};

  vk::SwapchainCreateInfoKHR createInfo = {
      .surface = surface,
      .minImageCount = imageCount,
      .imageFormat = capabilities.chooseFormat().format,
      .imageColorSpace = capabilities.chooseFormat().colorSpace,
      .imageExtent = extent,
      .imageArrayLayers = 1,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
      .imageSharingMode = sharingMode,
      .queueFamilyIndexCount = queueFamilyIndexCount,
      .pQueueFamilyIndices =
          sharingMode == vk::SharingMode::eExclusive
              ? nullptr
              : queueFamilyIndices,  // (nullptr when we are in exclusive mode)
      .preTransform = capabilities.capabilities.currentTransform,
      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
      .presentMode = capabilities.choosePresentMode(),
      .clipped = VK_TRUE,
      .oldSwapchain = nullptr,
  };

  return device.createSwapchainKHR(createInfo);
}

std::vector<vk::ImageView> createSwapChainImageViews(
    const vk::Device& device,
    const std::vector<vk::Image>& swapChainImages,
    const vk::Format& imageFormat) {
  std::vector<vk::ImageView> swapChainImageViews;
  swapChainImageViews.reserve(swapChainImages.size());

  for (const auto& swapChainImage : swapChainImages) {
    vk::ImageViewCreateInfo createInfo = {
        .image = swapChainImage,
        .viewType = vk::ImageViewType::e2D,
        .format = imageFormat,
        .components =
            {
                .r = vk::ComponentSwizzle::eIdentity,
                .g = vk::ComponentSwizzle::eIdentity,
                .b = vk::ComponentSwizzle::eIdentity,
                .a = vk::ComponentSwizzle::eIdentity,
            },
        .subresourceRange =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };
    swapChainImageViews.push_back(device.createImageView(createInfo));
  }

  return swapChainImageViews;
}

vk::ShaderModule createShaderModule(const vk::Device& device,
                                    const std::vector<uint8_t>& code) {
  vk::ShaderModuleCreateInfo createInfo = {
      .codeSize = code.size(),
      .pCode = reinterpret_cast<const uint32_t*>(code.data()),
  };
  return device.createShaderModule(createInfo);
}

vk::RenderPass createRenderPass(const vk::Device& device,
                                const vk::SurfaceFormatKHR& surfaceFormat) {
  vk::AttachmentDescription colorAttachment = {
      .format = surfaceFormat.format,
      .samples = vk::SampleCountFlagBits::e1,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
      .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
      .initialLayout = vk::ImageLayout::eUndefined,
      .finalLayout = vk::ImageLayout::ePresentSrcKHR,
  };
  vk::AttachmentReference colorAttachmentReference = {
      .attachment = 0,
      .layout = vk::ImageLayout::eColorAttachmentOptimal,
  };
  vk::SubpassDescription subpass = {
      .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
      .colorAttachmentCount = 1,
      .pColorAttachments = &colorAttachmentReference,
  };

  vk::RenderPassCreateInfo renderPassInfo = {
      .attachmentCount = 1,
      .pAttachments = &colorAttachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
  };
  return device.createRenderPass(renderPassInfo);
}

vk::PipelineLayout createPipelineLayout(const vk::Device& device) {
  vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {
      .setLayoutCount = 0,
      .pSetLayouts = nullptr,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = nullptr,
  };
  return device.createPipelineLayout(pipelineLayoutInfo);
}

vk::Pipeline createGraphicsPipeline(const vk::Device& device,
                                    const vk::Extent2D& swapChainExtent,
                                    const vk::RenderPass& renderPass,
                                    const vk::PipelineLayout& pipelineLayout) {
  auto vertShaderCode = readFile("shaders/shader.vert.spv");
  auto fragShaderCode = readFile("shaders/shader.frag.spv");

  auto vertShaderModule = createShaderModule(device, vertShaderCode);
  auto fragShaderModule = createShaderModule(device, fragShaderCode);

  vk::PipelineShaderStageCreateInfo vertShaderStageInfo = {
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = vertShaderModule,
      .pName = "main",
  };
  vk::PipelineShaderStageCreateInfo fragShaderStageInfo = {
      .stage = vk::ShaderStageFlagBits::eFragment,
      .module = fragShaderModule,
      .pName = "main",
  };
  vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                      fragShaderStageInfo};

  vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {
      .vertexBindingDescriptionCount = 0,
      .pVertexBindingDescriptions = nullptr,
      .vertexAttributeDescriptionCount = 0,
      .pVertexAttributeDescriptions = nullptr,
  };
  vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {
      .topology = vk::PrimitiveTopology::eTriangleList,
      .primitiveRestartEnable = VK_FALSE,
  };
  vk::PipelineViewportStateCreateInfo viewportState = {
      .viewportCount = 1,
      .scissorCount = 1,
  };
  vk::PipelineRasterizationStateCreateInfo rasterizer = {
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eBack,
      .frontFace = vk::FrontFace::eClockwise,
      .depthBiasEnable = VK_FALSE,
      .lineWidth = 1.0f,
  };
  vk::PipelineMultisampleStateCreateInfo multisampling = {
      .rasterizationSamples = vk::SampleCountFlagBits::e1,
      .sampleShadingEnable = VK_FALSE,
  };
  vk::PipelineColorBlendAttachmentState colorBlendAttachment = {
      .blendEnable = VK_FALSE,
      .colorWriteMask =
          vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };
  vk::PipelineColorBlendStateCreateInfo colorBlending = {
      .logicOpEnable = VK_FALSE,
      .attachmentCount = 1,
      .pAttachments = &colorBlendAttachment,
  };

  std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport,
                                                 vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dynamicState = {
      .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
      .pDynamicStates = dynamicStates.data(),
  };

  auto result = device
                    .createGraphicsPipelines(
                        nullptr,
                        {
                            {
                                .stageCount = 2,
                                .pStages = shaderStages,
                                .pVertexInputState = &vertexInputInfo,
                                .pInputAssemblyState = &inputAssembly,
                                .pViewportState = &viewportState,
                                .pRasterizationState = &rasterizer,
                                .pMultisampleState = &multisampling,
                                .pDepthStencilState = nullptr,
                                .pColorBlendState = &colorBlending,
                                .pDynamicState = &dynamicState,
                                .layout = pipelineLayout,
                                .renderPass = renderPass,
                                .subpass = 0,
                                .basePipelineHandle = nullptr,
                                .basePipelineIndex = -1,
                            },
                        })
                    .value[0];

  device.destroyShaderModule(fragShaderModule);
  device.destroyShaderModule(vertShaderModule);
  return result;
}

std::vector<vk::Framebuffer> createFrameBuffers(
    const vk::Device& device,
    const vk::Extent2D& swapChainExtent,
    const std::vector<vk::ImageView>& swapChainImageViews,
    const vk::RenderPass& renderPass) {
  std::vector<vk::Framebuffer> swapChainFramebuffers;
  swapChainFramebuffers.reserve(swapChainImageViews.size());

  for (const auto& imageView : swapChainImageViews) {
    vk::FramebufferCreateInfo framebufferInfo = {
        .renderPass = renderPass,
        .attachmentCount = 1,
        .pAttachments = &imageView,
        .width = swapChainExtent.width,
        .height = swapChainExtent.height,
        .layers = 1,
    };
    swapChainFramebuffers.push_back(device.createFramebuffer(framebufferInfo));
  }

  return swapChainFramebuffers;
}

void recordRenderCommands(const vk::Device& device,
                          const vk::Extent2D& swapChainExtent,
                          const vk::RenderPass& renderPass,
                          const vk::Pipeline& graphicsPipeline,
                          const vk::Framebuffer& framebuffer,
                          const vk::CommandBuffer& commandBuffer) {
  vk::CommandBufferBeginInfo beginInfo = {};
  commandBuffer.begin(beginInfo);

  vk::ClearValue clearColor = {{0.0f, 0.0f, 0.0f, 1.0f}};
  vk::RenderPassBeginInfo renderPassInfo = {
      .renderPass = renderPass,
      .framebuffer = framebuffer,
      .renderArea =
          {
              .offset = {0, 0},
              .extent = swapChainExtent,
          },
      .clearValueCount = 1,
      .pClearValues = &clearColor,
  };
  commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                             graphicsPipeline);

  vk::Viewport viewport = {
      .x = 0.0f,
      .y = 0.0f,
      .width = static_cast<float>(swapChainExtent.width),
      .height = static_cast<float>(swapChainExtent.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };
  commandBuffer.setViewport(0, viewport);
  vk::Rect2D scissor = {
      .offset = {0, 0},
      .extent = swapChainExtent,
  };
  commandBuffer.setScissor(0, scissor);

  commandBuffer.draw(3, 1, 0, 0);
  commandBuffer.endRenderPass();

  commandBuffer.end();
}

void draw(const vk::Device& device,
          const vk::SwapchainKHR& swapChain,
          const vk::Extent2D& swapChainExtent,
          const std::vector<vk::Framebuffer>& swapChainFramebuffers,
          const vk::Semaphore& acquireImage,
          const vk::Semaphore& renderFinished,
          const vk::Fence& inFlight,
          const vk::RenderPass& renderPass,
          const vk::Queue& graphicsQueue,
          const vk::Queue& presentQueue,
          const vk::Pipeline& graphicsPipeline,
          const vk::CommandBuffer& graphicsCommandBuffer) {
  device.waitForFences(inFlight, VK_TRUE, std::numeric_limits<uint64_t>::max());

  uint32_t imageIndex;
  device.acquireNextImageKHR(swapChain, std::numeric_limits<uint64_t>::max(),
                             acquireImage, nullptr, &imageIndex);

  graphicsCommandBuffer.reset({});
  recordRenderCommands(device, swapChainExtent, renderPass, graphicsPipeline,
                       swapChainFramebuffers[imageIndex],
                       graphicsCommandBuffer);

  vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eTopOfPipe};

  vk::SubmitInfo submitInfo = {
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &acquireImage,
      .pWaitDstStageMask = waitStages,
      .commandBufferCount = 1,
      .pCommandBuffers = &graphicsCommandBuffer,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &renderFinished,
  };

  device.resetFences(inFlight);
  graphicsQueue.submit(submitInfo, inFlight);

  vk::PresentInfoKHR presentInfo = {
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &renderFinished,
      .swapchainCount = 1,
      .pSwapchains = &swapChain,
      .pImageIndices = &imageIndex,
  };
  presentQueue.presentKHR(presentInfo);
}

int main() {
  if (glfwInit() != GLFW_TRUE) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }

  auto window = createGLFWWindow();

  try {
    auto vulkanInstance = createVulkanInstance();

    auto surface = createSurface(vulkanInstance, window);
    auto physicalDevice = pickPhysicalDevice(vulkanInstance, surface);
    auto queueIndices = RequiredQueueFamilyIndices(physicalDevice, surface);
    auto logicalDevice = createLogicalDevice(physicalDevice, queueIndices);
    auto queues = Queues(logicalDevice, queueIndices);
    auto capabilities = SwapChainCapabilities(physicalDevice, surface);
    auto swapChainExtent = capabilities.chooseExtent(window);
    auto swapChain = createSwapChain(logicalDevice, surface, capabilities,
                                     queueIndices, queues, swapChainExtent);
    auto swapChainImages = logicalDevice.getSwapchainImagesKHR(swapChain);
    auto swapChainImageViews = createSwapChainImageViews(
        logicalDevice, swapChainImages, capabilities.chooseFormat().format);

    auto renderPass =
        createRenderPass(logicalDevice, capabilities.chooseFormat());
    auto pipelineLayout = createPipelineLayout(logicalDevice);
    auto graphicsPipeline =
        createGraphicsPipeline(logicalDevice, capabilities.chooseExtent(window),
                               renderPass, pipelineLayout);

    auto swapChainFramebuffers = createFrameBuffers(
        logicalDevice, swapChainExtent, swapChainImageViews, renderPass);

    auto drawCommandPool = logicalDevice.createCommandPool(
        {.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
         .queueFamilyIndex = *queueIndices.graphicsFamily});
    auto drawCommandBuffer = logicalDevice.allocateCommandBuffers(
        {.commandPool = drawCommandPool,
         .level = vk::CommandBufferLevel::ePrimary,
         .commandBufferCount = 1})[0];

    auto acquireImageSemaphore = logicalDevice.createSemaphore({});
    auto renderFinishedSemaphore = logicalDevice.createSemaphore({});
    auto inFlightFence = logicalDevice.createFence(
        {.flags = vk::FenceCreateFlagBits::eSignaled});

    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      draw(logicalDevice, swapChain, swapChainExtent, swapChainFramebuffers,
           acquireImageSemaphore, renderFinishedSemaphore, inFlightFence,
           renderPass, queues.graphicsQueue, queues.presentQueue,
           graphicsPipeline, drawCommandBuffer);
    }

    logicalDevice.waitIdle();

    logicalDevice.destroyFence(inFlightFence);
    logicalDevice.destroySemaphore(renderFinishedSemaphore);
    logicalDevice.destroySemaphore(acquireImageSemaphore);

    logicalDevice.destroyCommandPool(drawCommandPool);

    for (const auto& framebuffer : swapChainFramebuffers) {
      logicalDevice.destroyFramebuffer(framebuffer); /*  */
    }
    logicalDevice.destroyPipeline(graphicsPipeline);
    logicalDevice.destroyPipelineLayout(pipelineLayout);
    logicalDevice.destroyRenderPass(renderPass);
    for (const auto& imageView : swapChainImageViews) {
      logicalDevice.destroyImageView(imageView);
    }
    logicalDevice.destroySwapchainKHR(swapChain);
    logicalDevice.destroy();
    vulkanInstance.destroySurfaceKHR(surface);
    vulkanInstance.destroy();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
