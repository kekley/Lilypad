use crate::{
    debug::{
        self, check_validation_layer_support, get_layer_names_and_pointers, setup_debug_messenger,
        ENABLE_VALIDATION_LAYERS,
    },
    swapchain::SwapchainSupportDetails,
    App,
};
use ash::{
    ext::debug_utils,
    khr::surface,
    vk::{
        self, make_api_version, ApplicationInfo, DeviceCreateInfo, InstanceCreateFlags,
        InstanceCreateInfo, PhysicalDevice, PhysicalDeviceFeatures, Queue, QueueFlags, SurfaceKHR,
        SwapchainKHR,
    },
    Device, Entry, Instance,
};
use std::{
    ffi::{CStr, CString},
    mem::ManuallyDrop,
    sync::Arc,
};
use vk_mem::{Allocator, AllocatorCreateInfo};
use winit::{
    event_loop::EventLoop,
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};
#[derive(Clone)]
pub struct VulkanContext {
    _entry: Entry,
    instance: Instance,
    debug_callback: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    surface: surface::Instance,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: Device,
    queues: AppQueues,
    queue_indices: QueueFamiliesIndices,
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        let device = self.device.clone();
        unsafe {
            self.surface.destroy_surface(self.surface_khr, None);

            device.destroy_device(None);
            match &self.debug_callback {
                Some((debug_instance, messenger)) => {
                    debug_instance.destroy_debug_utils_messenger(*messenger, None);
                }
                None => {}
            }
            self.instance.destroy_instance(None);
        };
    }
}

impl VulkanContext {
    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn surface(&self) -> &surface::Instance {
        &self.surface
    }

    pub fn surface_khr(&self) -> &vk::SurfaceKHR {
        &self.surface_khr
    }

    pub fn physical_device(&self) -> &vk::PhysicalDevice {
        &self.physical_device
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queues(&self) -> AppQueues {
        self.queues.clone()
    }

    pub fn queue_indices(&self) -> &QueueFamiliesIndices {
        &self.queue_indices
    }
}

impl VulkanContext {
    fn create_instance(entry: &Entry, window: &Window) -> Instance {
        let app_name = CString::new("Vulkan").unwrap();
        let engine_name = CString::new("Engine Name").unwrap();
        let app_info = ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(make_api_version(0, 0, 1, 0))
            .api_version(make_api_version(0, 1, 0, 0));

        let extension_names =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
                .unwrap();

        let mut extensions_names = extension_names.to_vec();

        if ENABLE_VALIDATION_LAYERS {
            extensions_names.push(debug_utils::NAME.as_ptr());
        }

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        let create_flags = InstanceCreateFlags::default();

        let mut instance_create_info = InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions_names)
            .flags(create_flags);

        if ENABLE_VALIDATION_LAYERS {
            check_validation_layer_support(entry);
            instance_create_info = instance_create_info.enabled_layer_names(&layer_names_ptrs);
        }

        unsafe { entry.create_instance(&instance_create_info, None).unwrap() }
    }
    fn get_required_device_extensions() -> [&'static CStr; 1] {
        [ash::khr::swapchain::NAME]
    }
    fn device_meets_reqs(
        instance: &Instance,
        surface: &surface::Instance,
        surface_khr: SurfaceKHR,
        device: PhysicalDevice,
    ) -> bool {
        let (graphics, present, transfer, compute) =
            Self::find_queue_families(&instance, &surface, surface_khr, device);
        let extension_support = Self::check_device_extension_support(instance, device);

        let is_swapchain_adequate = {
            let details = SwapchainSupportDetails::new(device, surface, surface_khr);
            !details.formats.is_empty() && !details.present_modes.is_empty()
        };

        let features = unsafe { instance.get_physical_device_features(device) };

        graphics.is_some() && present.is_some() && extension_support && is_swapchain_adequate
    }

    fn check_device_extension_support(instance: &Instance, device: vk::PhysicalDevice) -> bool {
        let required_extentions = Self::get_required_device_extensions();

        let extension_props = unsafe {
            instance
                .enumerate_device_extension_properties(device)
                .unwrap()
        };

        for required in required_extentions.iter() {
            let found = extension_props.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                required == &name
            });

            if !found {
                return false;
            }
        }

        true
    }

    fn pick_physical_device(
        instance: &Instance,
        surface: &surface::Instance,
        surface_khr: SurfaceKHR,
    ) -> (PhysicalDevice, QueueFamiliesIndices) {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let device = devices
            .into_iter()
            .find(|device| Self::device_meets_reqs(instance, surface, surface_khr, *device))
            .expect("Could not find suitable device");

        let props = unsafe { instance.get_physical_device_properties(device) };

        log::debug!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });

        let (graphics, present, transfer, compute) =
            Self::find_queue_families(&instance, &surface, surface_khr, device);

        let queue_family_indices = QueueFamiliesIndices {
            graphics_index: graphics.unwrap(),
            present_index: present.unwrap(),
            transfer_index: transfer,
            compute_index: compute,
        };
        (device, queue_family_indices)
    }

    fn find_queue_families(
        instance: &Instance,
        surface: &surface::Instance,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> (Option<u32>, Option<u32>, Option<u32>, Option<u32>) {
        let mut graphics = None;
        let mut present = None;
        let mut transfer = None;
        let mut compute = None;
        let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
            let index = index as u32;

            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
                graphics = Some(index);
            }

            let present_support = unsafe {
                surface
                    .get_physical_device_surface_support(device, index, surface_khr)
                    .unwrap()
            };
            if present_support && present.is_none() {
                present = Some(index);
            }

            if (family.queue_flags.contains(QueueFlags::TRANSFER)
                && !family.queue_flags.contains(QueueFlags::COMPUTE)
                && !family.queue_flags.contains(QueueFlags::GRAPHICS))
                && transfer.is_none()
            {
                transfer = Some(index);
            }
            if (family.queue_flags.contains(QueueFlags::COMPUTE)
                && !family.queue_flags.contains(QueueFlags::GRAPHICS))
                && compute.is_none()
            {
                compute = Some(index)
            }

            if graphics.is_some() && present.is_some() && compute.is_some() && transfer.is_some() {
                break;
            }
        }

        (graphics, present, transfer, compute)
    }
}

impl VulkanContext {
    pub fn get_mem_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        }
    }

    /// Find the first compatible format from `candidates`.
    pub fn find_supported_format(
        &self,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Option<vk::Format> {
        candidates.iter().cloned().find(|candidate| {
            let props = unsafe {
                self.instance
                    .get_physical_device_format_properties(self.physical_device, *candidate)
            };
            (tiling == vk::ImageTiling::LINEAR && props.linear_tiling_features.contains(features))
                || (tiling == vk::ImageTiling::OPTIMAL
                    && props.optimal_tiling_features.contains(features))
        })
    }
}

impl VulkanContext {
    pub fn new(window: &Window) -> Self {
        let entry = unsafe { Entry::load().expect("failed to create entry") };

        let instance = Self::create_instance(&entry, &window);

        let surface = surface::Instance::new(&entry, &instance);

        let debug_report_callback = setup_debug_messenger(&entry, &instance);

        let surface_khr = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .unwrap()
        };

        let (physical_device, queue_family_indices) =
            Self::pick_physical_device(&instance, &surface, surface_khr);

        let (device, queues) = Self::create_logical_device_and_queues(
            &instance,
            physical_device,
            queue_family_indices,
        );

        Self {
            device: device,
            queues: queues,
            instance: instance,
            _entry: entry,
            debug_callback: debug_report_callback,
            surface: surface,
            surface_khr: surface_khr,
            physical_device: physical_device,
            queue_indices: queue_family_indices,
        }
    }
}

impl VulkanContext {
    fn create_logical_device_and_queues(
        instance: &Instance,
        device: PhysicalDevice,
        queue_family_indices: QueueFamiliesIndices,
    ) -> (Device, AppQueues) {
        let queue_priorities = [1.0f32];

        let queue_create_infos = {
            let mut indices = vec![
                queue_family_indices.graphics_index,
                queue_family_indices.present_index,
            ];

            match queue_family_indices.transfer_index {
                Some(index) => {
                    indices.push(index);
                    println!("transfer: {:?}", index);
                }
                None => {}
            }
            match queue_family_indices.compute_index {
                Some(index) => {
                    indices.push(index);
                    println!("compute: {:?}", index);
                }
                None => {}
            }
            indices.dedup();

            indices
                .iter()
                .map(|index| {
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(*index)
                        .queue_priorities(&queue_priorities)
                })
                .collect::<Vec<_>>()
        };

        let device_extensions = Self::get_required_device_extensions();
        let device_extensions_ptrs = device_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let device_create_info = DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions_ptrs);

        let device = unsafe {
            instance
                .create_device(device, &device_create_info, None)
                .expect("failed to create logical device")
        };

        let graphics_queue =
            unsafe { device.get_device_queue(queue_family_indices.graphics_index, 0) };
        let present_queue =
            unsafe { device.get_device_queue(queue_family_indices.present_index, 0) };
        let transfer_queue = unsafe {
            match queue_family_indices.transfer_index {
                Some(index) => Some(device.get_device_queue(index, 0)),
                None => None,
            }
        };

        let compute_queue = unsafe {
            match queue_family_indices.compute_index {
                Some(index) => Some(device.get_device_queue(index, 0)),
                None => None,
            }
        };
        let queues = AppQueues {
            graphics_queue: graphics_queue,
            present_queue: present_queue,
            transfer_queue: transfer_queue,
            compute_queue: compute_queue,
        };
        (device, queues)
    }
}

#[derive(Clone, Copy)]
pub struct QueueFamiliesIndices {
    pub graphics_index: u32,
    pub present_index: u32,
    pub transfer_index: Option<u32>,
    pub compute_index: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct AppQueues {
    pub graphics_queue: Queue,
    pub present_queue: Queue,
    pub transfer_queue: Option<Queue>,
    pub compute_queue: Option<Queue>,
}
