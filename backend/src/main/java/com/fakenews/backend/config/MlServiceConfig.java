package com.fakenews.backend.config;

import java.net.http.HttpClient;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

@Configuration
public class MlServiceConfig {

    // Uses Spring Boot's pre-configured builder (not RestClient.builder()) so the JSON
    // converters pick up the app's Jackson settings, notably the snake_case naming
    // that maps the ML service's top_positive / models_available onto Java fields.
    @Bean
    public RestClient mlRestClient(RestClient.Builder builder, MlServiceProperties properties) {
        HttpClient httpClient = HttpClient.newBuilder()
                .connectTimeout(properties.connectTimeout())
                // Plain HTTP/1.1: skips the JDK client's h2c upgrade attempt, which
                // Python ASGI servers like uvicorn don't need to see.
                .version(HttpClient.Version.HTTP_1_1)
                .build();

        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory(httpClient);
        requestFactory.setReadTimeout(properties.readTimeout());

        return builder
                .baseUrl(properties.url())
                .requestFactory(requestFactory)
                .build();
    }
}
