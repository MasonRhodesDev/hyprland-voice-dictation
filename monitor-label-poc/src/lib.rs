use iced::widget::{container, text};
use iced::{Color, Element, Length, Task};
use iced_layershell::build_pattern::application;
use iced_layershell::reexport::{Anchor, KeyboardInteractivity, Layer};
use iced_layershell::settings::{LayerShellSettings, StartMode};
use iced_layershell::to_layer_message;
use smithay_client_toolkit::{
    delegate_output, delegate_registry,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
};
use std::thread;
use wayland_client::{
    globals::registry_queue_init, protocol::wl_output, Connection, QueueHandle,
};

struct MonitorManager {
    registry_state: RegistryState,
    output_state: OutputState,
    counter: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl OutputHandler for MonitorManager {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        if let Some(info) = self.output_state.info(&output) {
            let name = info.name.clone().unwrap_or_else(|| "Unknown Monitor".to_string());
            let monitor_name = name.clone();
            let id = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            println!("[MONITOR DETECTED #{}] Name: {}", id, monitor_name);
            println!("  - Description: {:?}", info.description);
            println!("  - Model: {}", info.model);
            println!("  - Make: {}", info.make);
            println!("  - Location: {:?}", info.location);
            println!("  - Spawning thread for monitor: {}", monitor_name);

            thread::spawn(move || {
                println!("[THREAD START #{}] Creating layer surface for: {}", id, monitor_name);
                let _ = run_monitor_label(id, monitor_name);
            });
        }
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }
}

impl ProvidesRegistryState for MonitorManager {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState];
}

delegate_output!(MonitorManager);
delegate_registry!(MonitorManager);

struct MonitorLabel {
    id: usize,
    monitor_name: String,
}

impl Default for MonitorLabel {
    fn default() -> Self {
        Self {
            id: 0,
            monitor_name: "Unknown".to_string(),
        }
    }
}

#[to_layer_message]
#[derive(Debug, Clone)]
enum Message {}

fn namespace(label: &MonitorLabel) -> String {
    format!("monitor-label-{}", label.monitor_name)
}

fn update(_label: &mut MonitorLabel, _message: Message) -> Task<Message> {
    Task::none()
}

fn view(label: &MonitorLabel) -> Element<'_, Message> {
    let display_text = format!("#{}: {}", label.id, label.monitor_name);
    let content = text(display_text)
        .size(20.0)
        .color(Color::WHITE);

    container(content)
        .width(Length::Fixed(200.0))
        .height(Length::Fixed(200.0))
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .style(|_theme: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(0.0, 0.0, 0.0, 0.8))),
            border: iced::Border {
                radius: 10.0.into(),
                ..Default::default()
            },
            ..Default::default()
        })
        .into()
}

fn style(_label: &MonitorLabel, theme: &iced::Theme) -> iced_layershell::Appearance {
    iced_layershell::Appearance {
        background_color: Color::TRANSPARENT,
        text_color: theme.palette().text,
    }
}

fn run_monitor_label(id: usize, monitor_name: String) -> Result<(), iced_layershell::Error> {
    let monitor_name_clone = monitor_name.clone();
    println!("[LAYER SURFACE #{}] Targeting screen: {}", id, monitor_name_clone);
    println!("  - Will display label: #{}: {}", id, monitor_name);

    application(namespace, update, view)
        .layer_settings(LayerShellSettings {
            size: Some((200, 200)),
            anchor: Anchor::Top | Anchor::Left,
            layer: Layer::Overlay,
            keyboard_interactivity: KeyboardInteractivity::None,
            margin: (10, 0, 0, 10),
            start_mode: StartMode::TargetScreen(monitor_name_clone),
            ..Default::default()
        })
        .style(style)
        .run_with(move || {
            println!("[ICED INIT #{}] Initializing app for monitor: {}", id, monitor_name);
            let label = MonitorLabel { id, monitor_name };
            (label, Task::none())
        })
}

pub fn run() -> anyhow::Result<()> {
    let conn = Connection::connect_to_env()?;
    let (globals, mut event_queue) = registry_queue_init(&conn)?;
    let qh = event_queue.handle();

    let mut manager = MonitorManager {
        registry_state: RegistryState::new(&globals),
        output_state: OutputState::new(&globals, &qh),
        counter: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    };

    event_queue.roundtrip(&mut manager)?;

    loop {
        event_queue.blocking_dispatch(&mut manager)?;
    }
}
