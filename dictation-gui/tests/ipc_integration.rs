use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_ipc_client_basic() {
    use dictation_gui::ipc::IpcClient;
    
    let client = IpcClient::new("/tmp/nonexistent_test_socket.sock".to_string());
    let has_no_stream = client.stream.is_none();
    assert!(has_no_stream);
}

#[tokio::test]
async fn test_control_client_basic() {
    use dictation_gui::control_ipc::ControlClient;
    
    let client = ControlClient::new("/tmp/nonexistent_control_socket.sock".to_string());
    let has_no_stream = client.stream.is_none();
    assert!(has_no_stream);
}

#[tokio::test]
async fn test_control_message_serialization() {
    use dictation_gui::control_ipc::ControlMessage;
    
    let msg = ControlMessage::TranscriptionUpdate {
        text: "integration test".to_string(),
        is_final: true,
    };
    
    let json = serde_json::to_string(&msg).unwrap();
    let parsed: ControlMessage = serde_json::from_str(&json).unwrap();
    
    match parsed {
        ControlMessage::TranscriptionUpdate { text, is_final } => {
            assert_eq!(text, "integration test");
            assert!(is_final);
        }
        _ => panic!("Wrong message type"),
    }
}

#[tokio::test]
async fn test_ipc_client_reconnect_attempt() {
    use dictation_gui::ipc::IpcClient;
    
    let mut client = IpcClient::new("/tmp/test_reconnect.sock".to_string());
    
    let result = client.connect().await;
    assert!(result.is_err());
    
    sleep(Duration::from_millis(50)).await;
}
