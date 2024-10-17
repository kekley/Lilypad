use std::ops::{Add, Rem};

use ash::vk::{Extent2D, Extent3D};
use glam::UVec2;

pub struct Position(UVec2);

impl Position {
    pub fn new(x: u32, y: u32) -> Self {
        Self {
            0: UVec2::new(x, y),
        }
    }
    pub fn x(&self) -> u32 {
        self.0.x
    }
    pub fn y(&self) -> u32 {
        self.0.y
    }
}
#[derive(Debug, Default, Clone, Copy)]
pub struct Extent(UVec2);

impl Add for Extent {
    type Output = Extent;

    fn add(self, rhs: Self) -> Self::Output {
        Extent(self.0 + rhs.0)
    }
}

impl Add<&Extent> for Extent {
    type Output = Extent;

    fn add(self, rhs: &Extent) -> Self::Output {
        self.add(*rhs)
    }
}

impl Add<&Extent> for &Extent {
    type Output = Extent;

    fn add(self, rhs: &Extent) -> Self::Output {
        (*self).add(*rhs)
    }
}

impl Add<Extent> for &Extent {
    type Output = Extent;

    fn add(self, rhs: Extent) -> Self::Output {
        Extent(self.0 + rhs.0)
    }
}

impl Rem for Extent {
    type Output = Extent;

    fn rem(self, rhs: Self) -> Self::Output {
        Extent(self.0 % rhs.0)
    }
}
impl Extent {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            0: UVec2::new(width, height),
        }
    }

    pub fn magnitude(&self) -> u32 {
        self.0.length_squared()
    }

    pub fn has_area(&self) -> bool {
        self.0.x > 0 && self.0.y > 0
    }

    pub fn width(&self) -> u32 {
        self.0.x
    }
    pub fn height(&self) -> u32 {
        self.0.y
    }
    pub fn to_vk_extent_2d(&self) -> Extent2D {
        Extent2D {
            width: self.width(),
            height: self.height(),
        }
    }
    pub fn to_vk_extent_3d(&self) -> Extent3D {
        Extent3D {
            width: self.width(),
            height: self.height(),
            depth: 1,
        }
    }
}

pub struct Rect {
    pub pos: Position,
    pub extent: Extent,
}
