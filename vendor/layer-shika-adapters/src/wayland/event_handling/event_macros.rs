#[macro_export]
macro_rules! bind_globals {
    ($global_list:expr, $queue_handle:expr, $(($interface:ty, $name:ident, $version:expr)),+) => {
        {
            $(
                let $name: $interface = $global_list.bind($queue_handle, $version, ())?;
            )+
            Ok::<($($interface,)+), LayerShikaError>(($($name,)+))
        }
    };
}
