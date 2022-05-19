mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"teixcbreno@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableXsrfProtection = false\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml