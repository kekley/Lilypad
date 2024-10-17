use std::io::{Cursor, Read};

use glam::{Affine2, Mat4, Quat, Vec2, Vec3, Vec4};
use uuid::Uuid;

use crate::{
    rects::{Position, Rect},
    renderer::{load_png, RenderContext, RenderObject, Texture2D},
    tile_set::TileSet,
};

pub struct TileInstance {
    pos: Position,
    data: TileData,
}

pub struct TileMap {
    tile_set: TileSet,
    width: u32,
    height: u32,
    tiles: Vec<TileInstance>,
}

impl TileMap {
    pub fn get_visible_tiles(&self, camera_rect: Rect) -> Vec<TileInstance> {
        todo!();
    }
}

pub struct TileData {
    tile_type: TileType,
    texture: Texture2D,
}

pub enum TileType {
    Empty,
    Solid,
    Liquid,
    Spike,
}

impl TileMap {
    pub fn draw(&self, context: &mut RenderContext) {
        for tile in &self.tiles {
            match tile.data.tile_type {
                TileType::Empty => continue,
                _ => {
                    let r: RenderObject = RenderObject {
                        obj_matrix: Affine2::from_scale_angle_translation(
                            Vec2::new(64.0, 64.0),
                            0.0,
                            Vec2::new(tile.pos.x() as f32 * 64.0, tile.pos.y() as f32 * 64.0),
                        ),
                        color: Vec4::splat(1.0),
                        texture: tile.data.texture.clone(),
                    };
                    context.opaque_objects.push(r);
                }
            }
        }
    }
    pub fn load_from_png(path: String, tile_set: TileSet) -> TileMap {
        let (image_data, extent) = load_png(path);
        println!("{}", image_data.len());
        assert!(image_data.len() as u32 == extent.width() * extent.height() * 3);
        let mut tiles = Vec::with_capacity((extent.width() * extent.height()) as usize);
        let mut image_cursor = Cursor::new(image_data);
        (0..extent.height()).for_each(|y| {
            (0..extent.width()).for_each(|x| {
                let mut pixel_buffer: [u8; 3] = [0; 3];
                image_cursor.read_exact(&mut pixel_buffer).expect("balls");
                println!("{:?}", pixel_buffer);
                let mut tile_type = TileType::Empty;
                let mut id: u32 = 0;
                match pixel_buffer {
                    [0, 0, 0] => {
                        tile_type = TileType::Solid;
                        id = 1
                    }
                    [255, 0, 0] => {
                        tile_type = TileType::Empty;
                        id = 0
                    }
                    [0, 255, 0] => {
                        tile_type = TileType::Liquid;
                        id = 5
                    }
                    [0, 0, 255] => {
                        println!("pixel: {:?}", pixel_buffer);
                        panic!()
                    }
                    _ => {
                        println!("pixel: {:?}", pixel_buffer);
                        panic!()
                    }
                }
                let data = TileData {
                    texture_index: id,
                    tile_type: tile_type,
                };
                println!("{},{}", x, y);

                tiles.push(TileInstance {
                    pos: Position::new(x, y),
                    data: data,
                })
            })
        });

        let map = TileMap {
            tile_set: tile_set,
            width: extent.width(),
            height: extent.height(),
            tiles: tiles,
        };
        map
    }
}
