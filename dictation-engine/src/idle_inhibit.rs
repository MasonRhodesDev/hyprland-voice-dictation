use anyhow::Result;
use tracing::info;
use zbus::zvariant::OwnedFd;
use zbus::{proxy, Connection};

#[proxy(
    interface = "org.freedesktop.login1.Manager",
    default_service = "org.freedesktop.login1",
    default_path = "/org/freedesktop/login1"
)]
trait LoginManager {
    fn inhibit(&self, what: &str, who: &str, why: &str, mode: &str) -> zbus::Result<OwnedFd>;
}

/// Holding the fd keeps the logind inhibit active. Dropping it (or the daemon
/// exiting) closes the fd and releases the inhibit automatically.
pub struct IdleInhibitor {
    _fd: OwnedFd,
}

pub async fn acquire(reason: &str) -> Result<IdleInhibitor> {
    let conn = Connection::system().await?;
    let proxy = LoginManagerProxy::new(&conn).await?;
    let fd = proxy.inhibit("idle:sleep", "voice-dictation", reason, "block").await?;
    info!("Idle/sleep inhibit acquired via logind");
    Ok(IdleInhibitor { _fd: fd })
}
