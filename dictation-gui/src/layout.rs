use taffy::prelude::*;
use taffy::{ResolveOrZero, TaffyTree};

pub struct OverlayLayout {
    taffy: TaffyTree<()>,
    root: NodeId,
    spectrum_node: NodeId,
    text_node: NodeId,
}

impl OverlayLayout {
    pub fn new(width: f32, initial_text_height: f32) -> Result<Self, taffy::TaffyError> {
        let mut taffy = TaffyTree::new();

        // Spectrum bars node (fixed height at top)
        let spectrum_node = taffy.new_leaf(Style {
            size: Size { width: length(width), height: length(50.0) },
            ..Default::default()
        })?;

        // Text node (dynamic height)
        let text_node = taffy.new_leaf(Style {
            size: Size { width: length(width), height: length(initial_text_height) },
            ..Default::default()
        })?;

        // Root container (column layout)
        let root = taffy.new_with_children(
            Style {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                size: Size { width: length(width), height: auto() },
                padding: Rect {
                    left: length(10.0),
                    right: length(10.0),
                    top: length(10.0),
                    bottom: length(10.0),
                },
                gap: Size { width: length(0.0), height: length(5.0) },
                ..Default::default()
            },
            &[spectrum_node, text_node],
        )?;

        Ok(Self { taffy, root, spectrum_node, text_node })
    }

    pub fn update_text_height(&mut self, height: f32) -> Result<(), taffy::TaffyError> {
        self.taffy.set_style(
            self.text_node,
            Style {
                size: Size {
                    width: length(
                        self.taffy.style(self.text_node)?.size.width.resolve_or_zero(None),
                    ),
                    height: length(height),
                },
                ..Default::default()
            },
        )
    }

    pub fn compute(
        &mut self,
        available_width: f32,
        available_height: f32,
    ) -> Result<(), taffy::TaffyError> {
        self.taffy.compute_layout(
            self.root,
            Size {
                width: AvailableSpace::Definite(available_width),
                height: AvailableSpace::Definite(available_height),
            },
        )
    }

    pub fn get_spectrum_rect(&self) -> Result<taffy::Layout, taffy::TaffyError> {
        self.taffy.layout(self.spectrum_node).copied()
    }

    pub fn get_text_rect(&self) -> Result<taffy::Layout, taffy::TaffyError> {
        self.taffy.layout(self.text_node).copied()
    }

    pub fn get_total_height(&self) -> Result<f32, taffy::TaffyError> {
        let layout = self.taffy.layout(self.root)?;
        Ok(layout.size.height)
    }
}
