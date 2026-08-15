use anyhow::Result;
use tracing::info;
use zbus::Connection;

/// Holding the fd keeps the logind inhibit active. Dropping it (or the daemon
/// exiting) closes the fd and releases the inhibit automatically.
pub struct IdleInhibitor {
    _inhibit: hypr_logind::Inhibitor,
}

pub async fn acquire(reason: &str) -> Result<IdleInhibitor> {
    let conn = Connection::system().await?;
    let proxy = hypr_logind::LogindManagerProxy::new(&conn).await?;
    let inhibit =
        hypr_logind::Inhibitor::acquire(&proxy, "idle:sleep", "voice-dictation", reason, "block")
            .await?;
    info!("Idle/sleep inhibit acquired via logind");
    Ok(IdleInhibitor { _inhibit: inhibit })
}
