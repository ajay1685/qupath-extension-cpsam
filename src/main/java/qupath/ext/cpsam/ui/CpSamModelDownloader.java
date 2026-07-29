package qupath.ext.cpsam.ui;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.function.DoubleConsumer;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import qupath.ext.cpsam.CpSamModel;

/**
 * Download TorchScript model archives (.zip) from GitHub releases
 * and extract them to a local directory.
 */
public class CpSamModelDownloader {

    private static final Logger logger = LoggerFactory.getLogger(CpSamModelDownloader.class);

    /** Buffer size for streaming copies (8 KB). */
    private static final int BUFFER_SIZE = 8192;

    private final Path cacheDir;

    public CpSamModelDownloader(Path cacheDir) {
        this.cacheDir = cacheDir;
    }

    // ── Public API ──────────────────────────────────────────────────────

    /**
     * Download a file from the given URL and verify the SHA-256 checksum (optional).
     * If the downloaded file is a .zip archive, it is automatically extracted to a
     * directory named after the archive (without .zip extension).
     */
    public DownloadResult downloadFromUrl(String url, String filename, String expectedSha256, DoubleConsumer progress) {
        try {
            // Derive target paths
            Path zipFile = cacheDir.resolve(filename);
            String dirName = stripZipExtension(filename);
            Path extractDir = cacheDir.resolve(dirName);

            // Check if already extracted and valid — use name-aware check so a wrong
            // model's .ts file (e.g. cpsam_v2_wrapper.ts in cpsam_wrapper/) isn't accepted
            if (CpSamModel.isValidModelDirForName(extractDir, dirName)) {
                logger.info("Model directory already valid: {}", extractDir);
                return DownloadResult.success(extractDir);
            }

            // Skip download if not requested
            boolean needDownload = !Files.isRegularFile(zipFile);
            if (!needDownload && expectedSha256 != null && !expectedSha256.isEmpty()) {
                String actual = sha256OfFile(zipFile);
                if (!expectedSha256.equalsIgnoreCase(actual.trim())) {
                    logger.warn("Existing zip checksum mismatch, re-downloading");
                    needDownload = true;
                }
            }

            // Download if needed
            if (needDownload) {
                Files.createDirectories(cacheDir);
                Path temp = cacheDir.resolve(filename + ".partial");

                logger.info("Downloading {} -> {}", url, zipFile);
                if (progress != null) progress.accept(0.0);

                downloadToFile(url, temp, progress);

                // Verify checksum if expected
                if (expectedSha256 != null && !expectedSha256.isEmpty()) {
                    String actual = sha256OfFile(temp);
                    if (!expectedSha256.equalsIgnoreCase(actual.trim())) {
                        Files.deleteIfExists(temp);
                        return DownloadResult.failure(
                                "Checksum verification failed. Expected: " + expectedSha256 + ", got: " + actual);
                    }
                    logger.info("SHA-256 verified: {}", actual);
                }

                // Atomically move to final zip location
                Files.move(temp, zipFile, StandardCopyOption.REPLACE_EXISTING);
            } else {
                if (progress != null) progress.accept(1.0);
            }

            // Extract zip
            unzip(zipFile, extractDir);

            // Verify .ts file exists
            if (!CpSamModel.isValidModelDir(extractDir)) {
                logger.error("Extracted directory does not contain a .ts file: {}", extractDir);
                return DownloadResult.failure("Downloaded archive did not contain a TorchScript model file (.ts)");
            }

            logger.info("Model extracted to: {}", extractDir);
            if (progress != null) progress.accept(1.0);
            return DownloadResult.success(extractDir);

        } catch (Exception e) {
            logger.error("Download failed: {}", e.getMessage(), e);
            return DownloadResult.failure("Download failed: " + e.getMessage());
        }
    }

    // ── HTTP download with progress ─────────────────────────────────────

    /**
     * Stream-download a URL to a file, reporting progress.
     */
    private void downloadToFile(String url, Path dest, DoubleConsumer progress) throws Exception {
        URL urlObj;
        try {
            urlObj = new URI(url).toURL();
        } catch (URISyntaxException e) {
            throw new Exception("Invalid URL: " + url, e);
        }

        var connection = urlObj.openConnection();
        connection.setConnectTimeout(15_000);
        connection.setReadTimeout(60_000);

        long contentLengthLong = connection.getContentLengthLong();
        int contentLength = contentLengthLong >= 0 && contentLengthLong <= Integer.MAX_VALUE
                ? (int) contentLengthLong
                : -1;

        try (ReadableByteChannel src = Channels.newChannel(connection.getInputStream());
             WritableByteChannel dst = Files.newByteChannel(dest, StandardOpenOption.WRITE,
                     StandardOpenOption.CREATE,
                     StandardOpenOption.TRUNCATE_EXISTING)) {

            ByteBuffer buf = ByteBuffer.allocate(BUFFER_SIZE);
            long totalRead = 0;

            while (true) {
                int r = src.read(buf);
                if (r <= 0) break;
                buf.flip();
                dst.write(buf);
                buf.clear();
                totalRead += r;

                if (contentLength > 0 && progress != null) {
                    double p = Math.min(1.0, (double) totalRead / contentLength);
                    progress.accept(p);
                }
            }
        }
    }

    // ── ZIP extraction ──────────────────────────────────────────────────

    /**
     * Unzip a .zip archive to a target directory.
     */
    private void unzip(Path zipFile, Path destination) throws Exception {
        if (!Files.exists(destination)) {
            Files.createDirectories(destination);
        }
        try (ZipInputStream zipIn = new ZipInputStream(
                new BufferedInputStream(new java.io.FileInputStream(zipFile.toFile())))) {
            ZipEntry entry;
            while ((entry = zipIn.getNextEntry()) != null) {
                Path filePath = destination.resolve(entry.getName());
                // Prevent ZIP slip — skip entries that escape the destination directory
                if (!filePath.normalize().startsWith(destination)) {
                    logger.warn("Skipping zip entry outside destination: {}", entry.getName());
                    zipIn.closeEntry();
                    continue;
                }
                if (entry.isDirectory()) {
                    Files.createDirectories(filePath);
                } else {
                    // Ensure parent dirs exist
                    Path parent = filePath.getParent();
                    if (parent != null && !Files.exists(parent)) {
                        Files.createDirectories(parent);
                    }
                    try (BufferedOutputStream bos = new BufferedOutputStream(
                            new FileOutputStream(filePath.toFile()))) {
                        byte[] bytesIn = new byte[4096];
                        int read;
                        while ((read = zipIn.read(bytesIn)) != -1) {
                            bos.write(bytesIn, 0, read);
                        }
                    }
                }
                zipIn.closeEntry();
            }
        }
    }

    /** Strip .zip extension to derive directory name. */
    private String stripZipExtension(String filename) {
        if (filename.toLowerCase().endsWith(".zip")) {
            return filename.substring(0, filename.length() - 4);
        }
        return filename;
    }

    // ── SHA-256 helper ──────────────────────────────────────────────────

    /** Compute SHA-256 of a file (single-pass streaming digest). */
    private String sha256OfFile(Path path) throws Exception {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            try (var dis = new DigestInputStream(Files.newInputStream(path), md);
                 var channel = Channels.newChannel(dis)) {
                ByteBuffer buf = ByteBuffer.allocate(BUFFER_SIZE * 4);
                while (channel.read(buf) > 0) {
                    buf.clear();
                }
            }
            return HexFormat.of().formatHex(md.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-256 not available", e);
        }
    }

    // ── Result DTO ──────────────────────────────────────────────────────

    /**
     * Immutable result of a download operation.
     */
    public static class DownloadResult {
        private final boolean success;
        private final Path path;
        private final String message;

        private DownloadResult(boolean success, Path path, String message) {
            this.success = success;
            this.path = path;
            this.message = message;
        }

        public static DownloadResult success(Path path) {
            return new DownloadResult(true, path, "OK");
        }

        public static DownloadResult failure(String message) {
            return new DownloadResult(false, null, message);
        }

        public boolean isSuccess() { return success; }
        public Path getPath() { return path; }
        public String getMessage() { return message; }

        @Override
        public String toString() {
            return success ? "Download OK: " + path : "Download FAILED: " + message;
        }
    }
}
