use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_control_server_client_communication() {
    use dictation_engine::control_ipc::{ControlMessage, ControlServer};

    let socket_path = "/tmp/test_control_integration_1.sock";
    let _ = std::fs::remove_file(socket_path);

    let mut server = ControlServer::new(socket_path).await.unwrap();

    let server_task = tokio::spawn(async move {
        sleep(Duration::from_millis(50)).await;
        server.try_accept().await;

        let result = server.broadcast(&ControlMessage::Ready).await;
        assert!(result.is_ok());

        server
    });

    sleep(Duration::from_millis(100)).await;

    let _server = server_task.await.unwrap();
    let _ = std::fs::remove_file(socket_path);
}


#[tokio::test]
async fn test_multiple_control_messages() {
    use dictation_engine::control_ipc::{ControlMessage, ControlServer};

    let socket_path = "/tmp/test_control_integration_2.sock";
    let _ = std::fs::remove_file(socket_path);

    let mut server = ControlServer::new(socket_path).await.unwrap();

    let msg1 = ControlMessage::Ready;
    let msg2 = ControlMessage::TranscriptionUpdate { text: "test".to_string(), is_final: false };

    assert!(server.broadcast(&msg1).await.is_ok());
    assert!(server.broadcast(&msg2).await.is_ok());

    let _ = std::fs::remove_file(socket_path);
}
