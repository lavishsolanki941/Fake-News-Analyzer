# NewsCheck backend (Spring Boot). Validates input, calls the ML service,
# stores prediction history in H2.
#
# Multi-stage build: stage 1 compiles the jar with Maven (needs the JDK +
# Maven, which the final image doesn't), stage 2 just runs that jar on a
# small JRE-only base. This keeps the image that actually ships much
# smaller than one image built to also hold the build tooling.
FROM maven:3.9-eclipse-temurin-17 AS build
WORKDIR /app

# Resolve dependencies in their own layer, cached as long as pom.xml is
# unchanged, so editing Java source doesn't re-download the internet.
COPY pom.xml .
RUN mvn -B dependency:go-offline

COPY src ./src
RUN mvn -B -DskipTests package

FROM eclipse-temurin:17-jre-jammy
WORKDIR /app
COPY --from=build /app/target/*.jar app.jar

# DB_PATH controls where the H2 file database lives (see application.yml).
# docker-compose mounts a named volume at /app/data so prediction history
# survives `docker compose down` / container recreation.
ENV DB_PATH=/app/data/newscheck-db

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "app.jar"]
