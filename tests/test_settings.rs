use config::Config;
use patentpick::settings::Settings;

# [test]
#[cfg(not(feature = "exclude_from_ci"))]
fn test_settings_new() {
    let settings = Settings::new("src/config.toml").unwrap();

    println!("{:?}", settings);
}


// Define another test function to check for errors
# [test]
fn test_settings_new_error(){
    // Use a mock config file path that does not exist
    let mock_config_file_path = "src/mock_config.toml";

    // Call the Config::builder() method with the mock file path
    let settings = Config::builder()
        .add_source(config::File::with_name(mock_config_file_path))
        .add_source(config::Environment::with_prefix("APP"))
        .build();

    let error = settings.unwrap_err();
    println!("{error}");

}
