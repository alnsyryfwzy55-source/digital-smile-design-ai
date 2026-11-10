// Digital Smile Design — offline Android app.
//
// The app is the same web app GitHub Pages serves (docs/), bundled as assets and served on the phone itself
// (http://localhost:8765), so it runs with no internet: the tooth model (ONNX), face landmarks (MediaPipe) and the
// simulation all run inside the WebView. Patient photos and cases stay on the phone (the WebView's IndexedDB).
//
// Native glue: camera permission, file export (share sheet) and printing the lab sheet.
import 'dart:collection';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:share_plus/share_plus.dart';

// Fixed port: the WebView keeps the cases per origin, so the origin must be the same on every launch.
const int kPort = 8765;

const Map<String, String> kMime = {
  'html': 'text/html; charset=utf-8',
  'js': 'text/javascript; charset=utf-8',
  'mjs': 'text/javascript; charset=utf-8',
  'css': 'text/css; charset=utf-8',
  'json': 'application/json; charset=utf-8',
  'webmanifest': 'application/manifest+json',
  'wasm': 'application/wasm',
  'onnx': 'application/octet-stream',
  'task': 'application/octet-stream',
  'png': 'image/png',
  'jpg': 'image/jpeg',
  'svg': 'image/svg+xml',
};

Future<void> startServer() async {
  final server = await HttpServer.bind(InternetAddress.loopbackIPv4, kPort, shared: true);
  server.listen((HttpRequest req) async {
    var path = Uri.decodeComponent(req.uri.path);
    if (path.isEmpty || path.endsWith('/')) path = '${path}index.html';
    try {
      final data = await rootBundle.load('assets/web$path');
      final bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
      final ext = path.split('.').last.toLowerCase();
      req.response.headers.set(HttpHeaders.contentTypeHeader, kMime[ext] ?? 'application/octet-stream');
      req.response.headers.set(HttpHeaders.cacheControlHeader, 'no-cache');
      req.response.contentLength = bytes.length;
      if (req.method != 'HEAD') req.response.add(bytes);
    } catch (_) {
      req.response.statusCode = HttpStatus.notFound;
    }
    await req.response.close();
  });
}

// Runs before the page's own scripts: Android's WebView has no Web Share and no window.print, so file downloads
// (blob links) and printing are handed to Flutter.
const String kBridge = r'''
(function () {
  function send(href, name) {
    fetch(href).then(function (r) { return r.blob(); }).then(function (b) {
      var fr = new FileReader();
      fr.onload = function () {
        window.flutter_inappwebview.callHandler('saveFile', name || 'file', b.type || '', String(fr.result).split(',')[1]);
      };
      fr.readAsDataURL(b);
    });
  }
  var click = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function () {
    if (this.hasAttribute('download') && /^(blob|data):/.test(this.href)) { send(this.href, this.getAttribute('download')); return; }
    return click.call(this);
  };
  window.print = function () { window.flutter_inappwebview.callHandler('print'); };
  window.__DSD_APP = 'android';
})();
''';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await startServer();
  runApp(const SmileApp());
}

class SmileApp extends StatelessWidget {
  const SmileApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'تصميم الابتسامة',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(colorSchemeSeed: const Color(0xFF0E7C66), useMaterial3: true),
        home: const WebShell(),
      );
}

class WebShell extends StatefulWidget {
  const WebShell({super.key});
  @override
  State<WebShell> createState() => _WebShellState();
}

class _WebShellState extends State<WebShell> {
  InAppWebViewController? _web;

  @override
  void initState() {
    super.initState();
    Permission.camera.request();
  }

  Future<void> _saveFile(List<dynamic> args) async {
    final name = (args.isNotEmpty ? args[0] : 'file').toString().replaceAll(RegExp(r'[\\/]'), '_');
    final mime = args.length > 1 ? args[1].toString() : '';
    final bytes = base64Decode(args.length > 2 ? args[2].toString() : '');
    final dir = await getTemporaryDirectory();
    final f = File('${dir.path}/$name');
    await f.writeAsBytes(bytes, flush: true);
    await Share.shareXFiles([XFile(f.path, mimeType: mime.isEmpty ? null : mime)]);
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, _) async {
        if (didPop) return;
        final w = _web;
        if (w != null && await w.canGoBack()) {
          await w.goBack();
        } else {
          SystemNavigator.pop();
        }
      },
      child: Scaffold(
        body: SafeArea(
          child: InAppWebView(
            initialUrlRequest: URLRequest(url: WebUri('http://localhost:$kPort/index.html')),
            initialUserScripts: UnmodifiableListView<UserScript>([
              UserScript(source: kBridge, injectionTime: UserScriptInjectionTime.AT_DOCUMENT_START),
            ]),
            initialSettings: InAppWebViewSettings(
              javaScriptEnabled: true,
              mediaPlaybackRequiresUserGesture: false,
              allowsInlineMediaPlayback: true,
              useHybridComposition: true,
              supportZoom: false,
              transparentBackground: false,
              cacheEnabled: true,
              domStorageEnabled: true,
              databaseEnabled: true,
            ),
            onWebViewCreated: (c) {
              _web = c;
              c.addJavaScriptHandler(handlerName: 'saveFile', callback: (args) async {
                await _saveFile(args);
                return true;
              });
              c.addJavaScriptHandler(handlerName: 'print', callback: (args) async {
                await c.printCurrentPage();
                return true;
              });
            },
            onPermissionRequest: (c, request) async {
              if (request.resources.contains(PermissionResourceType.CAMERA)) await Permission.camera.request();
              return PermissionResponse(resources: request.resources, action: PermissionResponseAction.GRANT);
            },
          ),
        ),
      ),
    );
  }
}
