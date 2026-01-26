from gymnasium.envs.registration import register as register

# Use HTTP-based environment (faster, more stable)
# Falls back to Selenium-based if HTTP server is not available
print("registering suika env (HTTP mode)")
register(
    id="SuikaEnv-v0",
    entry_point='suika_env.suika_http_env:SuikaBrowserEnv',
    max_episode_steps=100,
)
