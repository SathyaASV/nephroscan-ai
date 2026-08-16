import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;

public class NephroScanServer {

    public static void main(String[] args) throws IOException {

        int port = 8081;

        HttpServer server = HttpServer.create(
                new InetSocketAddress("localhost", port),
                0
        );

        // Health check
        server.createContext("/api/health", exchange -> {

            String response = """
                    {
                        "status": "online",
                        "service": "NephroScan AI",
                        "version": "3.0",
                        "message": "Java SE 23 backend is running"
                    }
                    """;

            sendJson(exchange, 200, response);
        });

        // AI analysis endpoint
        server.createContext("/api/analyze", exchange -> {

            if (!exchange.getRequestMethod().equalsIgnoreCase("POST")) {

                String response = """
                        {
                            "error": "Only POST requests are allowed"
                        }
                        """;

                sendJson(exchange, 405, response);
                return;
            }

            // Temporary response.
            // Later we will connect your actual AI model here.
            String response = """
                    {
                        "success": true,
                        "model": "NephroScan AI",
                        "architecture": "ResNet-50 + EfficientNet-B0",
                        "preprocessing": "Grayscale + CLAHE",
                        "segmentation": "U-Net",
                        "analysis": "Multi-Modal",
                        "message": "Scan received for AI analysis"
                    }
                    """;

            sendJson(exchange, 200, response);
        });

        server.start();

        System.out.println("----------------------------------------");
        System.out.println("      NephroScan AI Backend");
        System.out.println("----------------------------------------");
        System.out.println("Java Version : " + System.getProperty("java.version"));
        System.out.println("Server       : http://localhost:" + port);
        System.out.println("Health API   : http://localhost:" + port + "/api/health");
        System.out.println("Analyze API  : http://localhost:" + port + "/api/analyze");
        System.out.println("----------------------------------------");
        System.out.println("Backend is running...");
    }

    private static void sendJson(
            HttpExchange exchange,
            int statusCode,
            String response
    ) throws IOException {

        byte[] responseBytes =
                response.getBytes(StandardCharsets.UTF_8);

        exchange.getResponseHeaders()
                .set("Content-Type", "application/json; charset=UTF-8");

        exchange.getResponseHeaders()
                .set("Access-Control-Allow-Origin", "*");

        exchange.getResponseHeaders()
                .set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");

        exchange.getResponseHeaders()
                .set("Access-Control-Allow-Headers", "Content-Type");

        exchange.sendResponseHeaders(
                statusCode,
                responseBytes.length
        );

        try (OutputStream output =
                     exchange.getResponseBody()) {

            output.write(responseBytes);
        }
    }
}