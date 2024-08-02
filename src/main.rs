use std::clone;
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::device;
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::Queue;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags};
use vulkano::format::Format;
use vulkano::image::Image;
use vulkano::image::ImageUsage;
use vulkano::instance::InstanceExtensions;
#[allow(unused)]
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::swapchain::Surface;
use vulkano::swapchain::SurfaceInfo;
use vulkano::swapchain::Swapchain;
use vulkano::swapchain::SwapchainCreateInfo;
use vulkano::{Version, VulkanLibrary};
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, Event, StartCause, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowBuilder};
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_LUNARG_standard_validation"];

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

struct App {
    window: Arc<Window>,
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    physical_device: Arc<PhysicalDevice>,
    device: Arc<Device>,
    queues: HashMap<QueueFlags, Arc<Queue>>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<Image>>,
    memory_allocator: Arc<StandardMemoryAllocator>,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let window = Self::create_window(&event_loop);

        let required_extensions = Surface::required_extensions(&event_loop);
        let instance = Self::create_instance(required_extensions);

        let surface = Self::create_surface(&instance, &window);

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, index) =
            Self::select_physical_device(&instance, &surface, &device_extensions);

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, queue_map) =
            Self::create_device(physical_device.clone(), &device_extensions, index);

        let (swapchain, swapchain_images) =
            Self::create_swapchain(device.clone(), &surface, window.clone());

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        Self {
            window: window,
            instance: instance,
            surface: surface,
            physical_device: physical_device,
            device: device,
            queues: queue_map,
            swapchain: swapchain,
            swapchain_images: swapchain_images,
            memory_allocator: memory_allocator,
        }
    }

    fn create_window(event_loop: &EventLoop<()>) -> Arc<Window> {
        let window_builder = WindowBuilder::new()
            .with_active(true)
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
            .with_title("Vulkan");

        Arc::new(
            window_builder
                .build(&event_loop)
                .expect("Could not create window"),
        )
    }

    fn create_instance(required_extensions: InstanceExtensions) -> Arc<Instance> {
        let library = VulkanLibrary::new().expect("Could not find local vulkan library");

        Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                application_name: Some("Peanits".to_string()),
                max_api_version: Some(Version::major_minor(1, 3)),
                ..Default::default()
            },
        )
        .expect("Could not create instance")
    }

    fn create_surface(instance: &Arc<Instance>, window: &Arc<Window>) -> Arc<Surface> {
        Surface::from_window(instance.clone(), window.clone()).expect("Could not create surface")
    }
    fn select_physical_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface>,
        device_extensions: &DeviceExtensions,
    ) -> (Arc<PhysicalDevice>, u32) {
        instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
            .filter(|p| {
                p.supported_extensions().contains(device_extensions)
                    || p.api_version() >= Version::V1_3
            })
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.contains(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, surface).unwrap_or(false)
                    })
                    .map(|q| (p, q as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .expect("no device available")
    }

    fn get_exclusive_queue_index(
        physical_device: Arc<PhysicalDevice>,
        queue_flag: QueueFlags,
    ) -> Option<u32> {
        let index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(i, q)| {
                println!("Queue: {:?}, Flags: {:?}", i, q.queue_flags);
                if queue_flag == QueueFlags::TRANSFER {
                    q.queue_flags.contains(QueueFlags::TRANSFER)
                        && !q
                            .queue_flags
                            .intersects(QueueFlags::COMPUTE | QueueFlags::GRAPHICS)
                } else if queue_flag == QueueFlags::COMPUTE {
                    q.queue_flags.contains(QueueFlags::COMPUTE)
                        && !q.queue_flags.intersects(QueueFlags::GRAPHICS)
                } else {
                    q.queue_flags.contains(queue_flag)
                }
            });
        Some(index.expect("Could not find family") as u32)
    }

    fn create_device(
        physical_device: Arc<PhysicalDevice>,
        device_extensions: &DeviceExtensions,
        index: u32,
    ) -> (Arc<Device>, HashMap<QueueFlags, Arc<Queue>>) {
        let graphics_queue_index = index;
        let transfer_queue_index =
            Self::get_exclusive_queue_index(physical_device.clone(), QueueFlags::TRANSFER);

        let compute_queue_index =
            Self::get_exclusive_queue_index(physical_device.clone(), QueueFlags::COMPUTE);

        let mut queue_create_info_vec: Vec<QueueCreateInfo> = vec![];

        queue_create_info_vec.push(QueueCreateInfo {
            queue_family_index: index,
            ..Default::default()
        });

        match transfer_queue_index {
            Some(i) => {
                queue_create_info_vec.push(QueueCreateInfo {
                    queue_family_index: i,
                    ..Default::default()
                });
                println!("Transfer queue index: {}", i);
            }
            None => {}
        }
        match compute_queue_index {
            Some(i) => {
                queue_create_info_vec.push({
                    QueueCreateInfo {
                        queue_family_index: i,
                        ..Default::default()
                    }
                });
                println!("Compute queue index: {}", i);
            }
            None => {}
        }
        let create_info = DeviceCreateInfo {
            queue_create_infos: queue_create_info_vec,
            enabled_extensions: *device_extensions,
            ..Default::default()
        };

        let (device, queues) = Device::new(physical_device.clone(), create_info)
            .expect("Could not create logical device");

        let mut queue_map: HashMap<QueueFlags, Arc<Queue>> = HashMap::new();
        queues.for_each(|q| {
            if q.queue_family_index() == graphics_queue_index {
                queue_map.insert(QueueFlags::GRAPHICS, q);
            } else {
                match transfer_queue_index {
                    Some(i) => {
                        if q.queue_family_index() == i {
                            queue_map.insert(QueueFlags::TRANSFER, q.clone());
                        }
                    }
                    None => {}
                }
                match compute_queue_index {
                    Some(i) => {
                        if q.queue_family_index() == i {
                            queue_map.insert(QueueFlags::COMPUTE, q.clone());
                        }
                    }
                    None => {}
                }
                {}
            }
        });
        (device, queue_map)
    }

    fn create_swapchain(
        device: Arc<Device>,
        surface: &Arc<Surface>,
        window: Arc<Window>,
    ) -> (
        Arc<vulkano::swapchain::Swapchain>,
        Vec<Arc<vulkano::image::Image>>,
    ) {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(surface, SurfaceInfo::default())
            .unwrap();

        let image_formats = device
            .physical_device()
            .surface_formats(surface, SurfaceInfo::default())
            .unwrap();

        let image_format = image_formats
            .iter()
            .enumerate()
            .find(|(i, f)| f.0 == Format::R8G8B8A8_SRGB)
            .expect("Could not find 8 bit rgba color format")
            .1
             .0;

        Swapchain::new(
            device,
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .expect("Could not create swapchain")
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let app = App::new(&event_loop);
    event_loop.run(|event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        _ => (),
    });
}
