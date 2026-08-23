use anyhow::Result;
use tracing::info;
use zbus::Connection;

/// Holding the fd keeps the logind inhibit active. Dropping it (or the daemon
/// exiting) closes the fd and releases the inhibit automatically.
pub struct IdleInhibitor {
    _inhibit: logind_session::Inhibitor,
}

pub async fn acquire(reason: &str) -> Result<IdleInhibitor> {
    let conn = Connection::system().await?;
    let proxy = logind_session::LogindManagerProxy::new(&conn).await?;
    let inhibit =
        logind_session::Inhibitor::acquire(&proxy, "idle:sleep", "voice-dictation", reason, "block")
            .await?;
    info!("Idle/sleep inhibit acquired via logind");
    Ok(IdleInhibitor { _inhibit: inhibit })
}
