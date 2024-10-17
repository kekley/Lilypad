use ash::vk::{
    BufferUsageFlags, CommandBuffer, DescriptorSet, DescriptorSetLayout, DescriptorType, Extent2D,
    Filter, Format, ImageAspectFlags, ImageLayout, ImageTiling, ImageUsageFlags, ImageView,
    Pipeline, PipelineBindPoint, PipelineLayout, PushConstantRange, RenderPass, SampleCountFlags,
    Sampler, SamplerCreateInfo, ShaderStageFlags, VertexInputAttributeDescription,
    VertexInputBindingDescription, VertexInputRate,
};
use glam::{Affine2, Affine3A, Mat3, Mat3A, Mat4, Vec2, Vec3, Vec4};
use inline_spirv::include_spirv;
use load_file::load_bytes;
use png::{Decoder, Transformations};
use uuid::Uuid;
use std::{
    char::MAX, collections::{HashMap, HashSet}, ffi::CString, fmt, io::{Cursor, Read, Seek, Write}, mem::transmute, path::Path, sync::Arc
};
use vk_mem::{Alloc, Allocator};

use crate::{
    descriptors::{DescriptorAllocator, DescriptorWriter, PoolSizeRatio}, pipeline_builder::{AsPushConstantRange, GraphicsPipelineBuilder, VertexAttributes}, rects::{Extent, Rect}, render_frame::RenderFrame, resource_manager::{self, AllocatedBuffer, AllocatedImage, ResourceManager}, shaders::{
        DescriptorLayoutBuilder, Shader, SPRITE_FRAGMENT_SHADER_CODE, SPRITE_VERTEX_SHADER_CODE,
    }, tile_map::TileMap, HEIGHT, WIDTH
};
const TEXTURE_ARRAY_SIZE: u32 = 512;
pub const SPRITE_VERTICES: [SpriteVertex; 6] = [
    SpriteVertex {
        pos: Vec2::new(0.0, 1.0),
        uv: Vec2::new(0.0, 1.0),
    },
    SpriteVertex {
        pos: Vec2::new(1.0, 0.0),
        uv: Vec2::new(1.0, 0.0),
    },
    SpriteVertex {
        pos: Vec2::new(0.0, 0.0),
        uv: Vec2::new(0.0, 0.0),
    },
    SpriteVertex {
        pos: Vec2::new(0.0, 1.0),
        uv: Vec2::new(0.0, 1.0),
    },
    SpriteVertex {
        pos: Vec2::new(1.0, 1.0),
        uv: Vec2::new(1.0, 1.0),
    },
    SpriteVertex {
        pos: Vec2::new(1.0, 0.0),
        uv: Vec2::new(1.0, 0.0),
    },
];

#[derive(Default, Debug)]
#[repr(C)]
pub struct SpritePushConstants {
    sprite_color: Vec4,
    texture_index: u32,
}

impl AsPushConstantRange for SpritePushConstants {
    fn push_constant_range(&self) -> PushConstantRange {
        PushConstantRange::default()
            .offset(0)
            .size(std::mem::size_of::<SpritePushConstants>() as u32)
            .stage_flags(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT)
    }
}

impl SpritePushConstants {
    pub fn as_bytes(self) -> [u8; 32] {
        unsafe { transmute::<SpritePushConstants, [u8; 32]>(self) }
    }
}
#[derive(Debug)]
pub struct RenderObject {
    pub obj_matrix: Affine2,
    pub color: Vec4,
    pub texture: Texture2D,
}
#[derive(Default, Debug)]
pub struct RenderContext {
    pub opaque_objects: Vec<RenderObject>,
    pub transparent_objects: Vec<RenderObject>
}

impl RenderContext {
    pub fn num_objects(&self)->u32{
        (self.opaque_objects.len()+self.transparent_objects.len()) as u32
    }
}
#[derive(Default,Copy,Clone, Debug)]
#[repr(C)]
pub struct SpriteVertex {
    pos: Vec2,
    uv: Vec2,
}

impl SpriteVertex {
    pub fn as_bytes(self) -> [u8; 16] {
        let bytes = unsafe { std::mem::transmute::<SpriteVertex, [u8; 16]>(self) };
        bytes
    }
}

impl VertexAttributes for SpriteVertex {
    fn input_attributes(&self) -> Vec<ash::vk::VertexInputAttributeDescription> {
        let pos = VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(Format::R32G32_SFLOAT)
            .offset(std::mem::offset_of!(SpriteVertex, pos) as u32);
        let uv = VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(Format::R32G32_SFLOAT)
            .offset(std::mem::offset_of!(SpriteVertex, uv) as u32);
        vec![pos, uv]
    }

    fn input_bindings(&self) -> Vec<ash::vk::VertexInputBindingDescription> {
        vec![VertexInputBindingDescription::default()
            .binding(0)
            .input_rate(VertexInputRate::VERTEX)
            .stride(std::mem::size_of::<SpriteVertex>() as u32)]
    }
}

pub fn load_png(path: String) -> (Vec<u8>, Extent) {
    let bytes = load_bytes!(&path);
    let cursor = Cursor::new(bytes);
    let mut decoder = Decoder::new(cursor);
    decoder.set_transformations(Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().unwrap();
    
    let info = reader.info();
    let image_dimensions =Extent::new(info.width, info.height);

    let mut image_data = Vec::new();

    image_data.resize(reader.output_buffer_size(), 0);
    reader.next_frame(&mut image_data).unwrap();
    (image_data, image_dimensions)
}
pub struct Renderer2D {
    device: ash::Device,
    render_context: RenderContext,
    camera_descriptor_layout: DescriptorSetLayout,
    texture_descriptor_layout: DescriptorSetLayout,
    pipeline_layout: PipelineLayout,
    pipeline: Pipeline,
    nearest_sampler: Sampler,
    linear_sampler: Sampler,
    loaded_images: HashMap<Uuid,AllocatedImage>,
    missing_texture: Texture2D,
    global_descriptor_allocator: DescriptorAllocator,
    resource_manager: Arc<ResourceManager>,
}

impl Renderer2D {
    pub fn add_texture(&mut self, image: AllocatedImage,texture_name: String) -> Texture2D {
        let new_id = Uuid::new_v4();
        let extent = image.extent;
        self.loaded_images.insert(new_id, image);
        Texture2D { texture_id: new_id, extent: extent, name: texture_name }
    }
}

impl Renderer2D {
    pub fn new(
        device: &ash::Device,
        resource_manager: Arc<ResourceManager>,
        render_pass: RenderPass,
    ) -> Self {
        let vert_shader = Shader::new(&device, ShaderStageFlags::VERTEX, SPRITE_VERTEX_SHADER_CODE);
        let frag_shader = Shader::new(
            &device,
            ShaderStageFlags::FRAGMENT,
            SPRITE_FRAGMENT_SHADER_CODE,
        );
        let mut camera_matrix_descriptor_layout_builder = DescriptorLayoutBuilder::new();
        camera_matrix_descriptor_layout_builder.add_binding(
            0,
            ShaderStageFlags::VERTEX,
            1,
            DescriptorType::UNIFORM_BUFFER,
        );
        let mut textures_descriptor_layout_builder = DescriptorLayoutBuilder::new();

        textures_descriptor_layout_builder.add_binding(
            0,
            ShaderStageFlags::FRAGMENT,
            1,
            DescriptorType::SAMPLER,
        );
        textures_descriptor_layout_builder.add_binding(
            1,
            ShaderStageFlags::FRAGMENT,
            TEXTURE_ARRAY_SIZE,
            DescriptorType::SAMPLED_IMAGE,
        );

        let camera_descriptor_layout =
            camera_matrix_descriptor_layout_builder.build(&device, ShaderStageFlags::VERTEX);

        let texture_descriptor_layout =
            textures_descriptor_layout_builder.build(&device, ShaderStageFlags::FRAGMENT);

        let mut pipeline_builder = GraphicsPipelineBuilder::new(
            vert_shader,
            frag_shader,
            Some(Box::new(SpriteVertex::default())),
            Some(Box::new(SpritePushConstants::default())),
        );
        pipeline_builder.add_descriptor_layout(camera_descriptor_layout);
        pipeline_builder.add_descriptor_layout(texture_descriptor_layout);

        pipeline_builder.enable_msaa();
        pipeline_builder.enable_alpha_blending();

        let (pipeline_layout, pipeline) = pipeline_builder.build(&device, render_pass);

        let pool_ratios = [
            PoolSizeRatio {
                descriptor_type: DescriptorType::SAMPLED_IMAGE,
                ratio: TEXTURE_ARRAY_SIZE as f32,
            },
            PoolSizeRatio {
                descriptor_type: DescriptorType::SAMPLER,
                ratio: 2.0,
            },
            PoolSizeRatio {
                descriptor_type: DescriptorType::UNIFORM_BUFFER,
                ratio: 1.0,
            },
        ];
        let mut descriptor_allocator = DescriptorAllocator::new(&device, 1, &pool_ratios);

        let texture_set = descriptor_allocator.allocate(texture_descriptor_layout);

        let (image_data, dimensions) = load_png("../textures/0.png".to_string());
        let format =Format::R8G8B8A8_UNORM;

        let image = resource_manager.create_image_from_data(
            dimensions,
            1,
            SampleCountFlags::TYPE_1,
            format,
            ImageTiling::OPTIMAL,
            ImageUsageFlags::SAMPLED,
            ImageAspectFlags::COLOR,
            image_data.as_ptr(),
            image_data.len(),
        );


        let sampler_create_info = SamplerCreateInfo::default()
            .mag_filter(Filter::NEAREST)
            .min_filter(Filter::NEAREST);
        let nearest_sampler = unsafe { device.create_sampler(&sampler_create_info, None).unwrap() };
        let sampler_create_info = SamplerCreateInfo::default()
            .mag_filter(Filter::LINEAR)
            .min_filter(Filter::LINEAR);
        let linear_sampler = unsafe { device.create_sampler(&sampler_create_info, None).unwrap() };
        let mut writer = DescriptorWriter::new();

        writer.write_image(
            0,
            ImageView::null(),
            nearest_sampler,
            ImageLayout::default(),
            DescriptorType::SAMPLER,
        );

        let image_views = (0..TEXTURE_ARRAY_SIZE)
            .map(|_f| image.view)
            .collect();

        let mut textures: HashMap<Uuid,AllocatedImage> = HashMap::default();
        let missing_texture_id = Uuid::new_v4();
        let missing_texture = Texture2D { texture_id:missing_texture_id,extent:image.extent,name: "missing_texture".to_string()};
        textures.insert(missing_texture_id, image);

        writer.write_texture_array(
            1,
            image_views,
            ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            DescriptorType::SAMPLED_IMAGE,
        );
        writer.update(device, texture_set);

        Self {
            resource_manager: resource_manager.clone(),
            nearest_sampler: nearest_sampler,
            linear_sampler: linear_sampler,
            camera_descriptor_layout: camera_descriptor_layout,
            texture_descriptor_layout: texture_descriptor_layout,
            loaded_images: textures,
            missing_texture: missing_texture,
            device: device.clone(),
            render_context: RenderContext::default(),
            pipeline_layout: pipeline_layout,
            pipeline: pipeline,
            global_descriptor_allocator: descriptor_allocator,
        }
    }
}

impl Renderer2D {
    pub fn add_sprite_sheet(&mut self,path: String,tile_size: Extent, padding: Option<Extent>) -> Vec<Texture2D> {
        let images = load_sprite_sheet(
            &self.resource_manager,
            path,
            tile_size,
            padding
        );
        let mut return_textures = Vec::with_capacity(images.len());
        assert!(images.len() <= (TEXTURE_ARRAY_SIZE - 1) as usize);
        images
            .into_iter()
            .enumerate()
            .for_each(|(i, image)| {
                let tex = self.add_texture(image, format!("tile_map_tile_{}",i));
                return_textures.push(tex);
            });
            return_textures
    }

}


impl Renderer2D {
    pub fn render_frame(& mut self){

    }
}

impl Renderer2D {
    pub fn draw_sprites(&mut self, cmd: &CommandBuffer, frame_data: &mut RenderFrame) {
        let mut vertex_buffer = Vec::with_capacity(self.render_context.num_objects() as usize * SPRITE_VERTICES.len());
        let mut textures = Vec::with_capacity(self.render_context.num_objects() as usize);
        for render_object in &self.render_context.opaque_objects {
             SPRITE_VERTICES.iter().for_each(|vertex| { vertex_buffer.push(SpriteVertex { pos: render_object.obj_matrix.transform_point2(vertex.pos), uv: vertex.uv })});
             match self.loaded_images.get(&render_object.texture.texture_id) {
                Some(image) => {textures.push(image.view)},
                None => {textures.push(self.loaded_images.get(&self.missing_texture.texture_id).unwrap().view)},
             }
            }
        let camera_matrix =
            glam::Mat4::orthographic_rh(0.0, WIDTH as f32, 0.0 as f32, HEIGHT as f32, 0.0, 1.0);
        let as_bytes = unsafe { transmute::<Mat4, [u8; 64]>(camera_matrix) };
        let camera_data_buffer = self.resource_manager.create_and_upload_buffer(
            size_of::<Mat4>() as u32,
            BufferUsageFlags::UNIFORM_BUFFER,
            vk_mem::MemoryUsage::Auto,
            as_bytes.as_ptr(),
        );

        let mut writer = DescriptorWriter::new();
        writer.write_buffer(
            0,
            camera_data_buffer.buffer,
            size_of::<Mat4>() as u64,
            0,
            DescriptorType::UNIFORM_BUFFER,
        );

        let camera_descriptor_set = frame_data.descriptor_set(self.camera_descriptor_layout);
        writer.update(&self.device, camera_descriptor_set);

        writer.clear();
        writer.write_image(1, ImageView::null(), self.nearest_sampler, ImageLayout::default(), DescriptorType::SAMPLER);
        writer.write_texture_array(1, textures, ImageLayout::SHADER_READ_ONLY_OPTIMAL, DescriptorType::SAMPLED_IMAGE);
        let texture_descriptor_set = frame_data.descriptor_set(self.texture_descriptor_layout);
        writer.update(&self.device, texture_descriptor_set);

        let data_ptr = vertex_buffer.as_ptr() as *const u8;
        
        let allocated_vertex_buffer = self.resource_manager.create_and_upload_buffer((vertex_buffer.len()*size_of::<SpriteVertex>()) as u32, BufferUsageFlags::VERTEX_BUFFER, vk_mem::MemoryUsage::Auto, data_ptr);
        unsafe {
            self.device
                .cmd_bind_pipeline(*cmd, PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device
                .cmd_bind_vertex_buffers(*cmd, 0, &[allocated_vertex_buffer.buffer], &[0]);
            self.device.cmd_bind_descriptor_sets(
                *cmd,
                PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[camera_descriptor_set],
                &[],
            );
            self.device.cmd_bind_descriptor_sets(
                *cmd,
                PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                1,
                &[texture_descriptor_set],
                &[],
            );
        };




        frame_data.push_buffer(camera_data_buffer);
        frame_data.push_buffer(allocated_vertex_buffer);

        self.render_context.opaque_objects.clear();
        self.render_context.transparent_objects.clear();
    }

    pub fn draw_loaded_textures(&mut self, ) {

        let mut render_objects = Vec::default();
        let mut x = 0.0;
        let mut y = 0.0;
        let mut i = 0;
        for texture in &self.loaded_images {
            let r = RenderObject {
                obj_matrix: Affine2::from_scale_angle_translation(Vec2::new(64.0, 64.0), 0.0, Vec2::new(x*64.0, y * 64.0 )),
                color: Vec4::new(1.0,0.0,0.0,1.0),
                texture: Texture2D { texture_id: *texture.0, extent: texture.1.extent, name: "".to_string() },
                
            };
            i = i+1;
            if x < 30.0 {
                x = x + 1.0;
            } else {
                x = 0.0;
                y = y + 1.0;
            }
            render_objects.push(r);
        }

    }

    pub fn draw_tile_map(&mut self, map: &TileMap){
        map.draw(&mut self.render_context);
    }
}

#[derive(Debug,Default,Clone)]
pub struct Texture2D {
    texture_id: Uuid,
    extent: Extent,
    name: String
}



impl Drop for Renderer2D {
    fn drop(&mut self) {
        unsafe {
            let a: () = self.device.device_wait_idle().unwrap();

            self.device.destroy_sampler(self.linear_sampler, None);
            self.device.destroy_sampler(self.nearest_sampler, None);
            self.device
                .destroy_descriptor_set_layout(self.camera_descriptor_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.texture_descriptor_layout, None);

            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
        };
    }
}

fn load_sprite_sheet(
    resource_manager: &ResourceManager,
    path: String,
    tile_size: Extent,
    padding: Option<Extent>,
) -> Vec<AllocatedImage> {
    let (image, image_size) = load_png(path);

    let rows : u32;
    let cols : u32;
    let size_with_padding : Extent;
    match &padding {
        Some(p)=>{
            size_with_padding = tile_size+p;
            assert!((image_size.width()+1) % size_with_padding.width() ==0);
            assert!((image_size.height()+1) % size_with_padding.height() ==0);
            rows = (image_size.width() + 1) / size_with_padding.width();
            cols = (image_size.height()+1) / size_with_padding.width();
        },
        None =>{
            assert!(image_size.width()%tile_size.width()==0);
            assert!(image_size.height()%tile_size.height()==0);
            size_with_padding = tile_size;
            rows = image_size.width()/tile_size.width();
            cols = image_size.height()/tile_size.height();
        }
    }    
    let bit_depth = 4;

    
    let byte_size_of_tile = (tile_size.width()*tile_size.height() * bit_depth) as usize;
    let byte_size_of_horizontal_tile_line = (tile_size.width() * bit_depth) as usize;
    let byte_size_of_horizontal_padding = ((size_with_padding.width()-tile_size.width())*bit_depth) as usize;


    let byte_size_of_image_stride = (image_size.width() * bit_depth) as usize;

    let tile_buffer: Vec<u8> = vec![0;byte_size_of_tile];
    let mut line_buffer: Vec<u8> = vec![0;byte_size_of_horizontal_tile_line as usize];
    let mut image_cursor: Cursor<Vec<u8>> = Cursor::new(image);
    let mut tile_cursor: Cursor<Vec<u8>> = Cursor::new(tile_buffer);

    let mut images= Vec::with_capacity((rows*cols) as usize);

    let mut i = 1;
    (1..cols+1).for_each(|y| {(1..rows+1).for_each(|x|{
        (0..tile_size.height()).for_each(|j| {
                image_cursor.read_exact(&mut line_buffer).expect(
                    format!("read past image file. tile num:{:?} line num {:?}  ",i, j).as_str(),
                );
                image_cursor
                    .seek_relative(byte_size_of_image_stride as i64 - byte_size_of_horizontal_tile_line as i64)
                    .expect(
                        format!("seeked too far back. tile num:{:?} line num {:?}  ", i, j)
                            .as_str(),
                    );
                tile_cursor.write(&line_buffer).expect(
                    format!(
                        "no bytes to write from in line buffer. tile num:{:?} line num {:?}  ",
                        i, j
                    )
                    .as_str(),
                );
            });
            
            tile_cursor.flush().unwrap();

            let tile = tile_cursor.get_ref();
            let image = resource_manager.create_image_from_data(
                Extent::new(tile_size.width(), tile_size.height()),
                1,
                SampleCountFlags::TYPE_1,
                Format::R8G8B8A8_UNORM,
                ImageTiling::OPTIMAL,
                ImageUsageFlags::SAMPLED,
                ImageAspectFlags::COLOR,
                tile.as_ptr(),
                byte_size_of_tile,
            );
            let x_offset = (x-1)*(byte_size_of_horizontal_tile_line as u32+byte_size_of_horizontal_padding as u32);
            let y_offset = (y-1)*size_with_padding.height()*byte_size_of_image_stride as u32;
            image_cursor.seek(std::io::SeekFrom::Start((x_offset+y_offset) as u64)).expect(
                format!(
                    "could not seek to next tile. cursor_position: {:?}, offset: {:?} tile num:{:?} ",
                    image_cursor.position(),
                    x_offset+y_offset,
                    i,
                    
                )
                .as_str(),
            );
            tile_cursor
                .rewind()
                .expect(format!("could not rewind tile cursor. tile num:{:?}  ", i).as_str());
                i=i+1;

            images.push(image);
    })});

    images
}
