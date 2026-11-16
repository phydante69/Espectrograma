import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, Line
from kivy.uix.widget import Widget
from kivy.utils import platform

SR = 44100      # frecuencia de muestreo (Hz)
N = 2048        # muestras por análisis (resolución ≈ SR/N ≈ 21.5 Hz)
FMAX = 8000     # frecuencia máxima a mostrar (Hz)

# ---------- Fuente de audio ----------
if platform == "android":
    from jnius import autoclass
    from android.permissions import request_permissions, Permission

    AudioRecord = autoclass("android.media.AudioRecord")
    AudioSource = autoclass("android.media.MediaRecorder$AudioSource")
    AudioFormat = autoclass("android.media.AudioFormat")

    class Mic:
        def __init__(self):
            min_buf = AudioRecord.getMinBufferSize(
                SR, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            self.rec = AudioRecord(
                AudioSource.MIC, SR, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, max(min_buf, N * 4))
            self.rec.startRecording()
            self.buf = bytearray(N * 2)   # 16 bits = 2 bytes por muestra

        def read(self):
            self.rec.read(self.buf, 0, len(self.buf))
            return np.frombuffer(bytes(self.buf), dtype="<i2").astype(np.float32)

        def stop(self):
            self.rec.stop()
            self.rec.release()
else:
    # Para probar en el PC: pip install sounddevice
    import sounddevice as sd

    class Mic:
        def __init__(self):
            self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16")
            self.stream.start()

        def read(self):
            data, _ = self.stream.read(N)
            return data[:, 0].astype(np.float32)

        def stop(self):
            self.stream.stop()

# ---------- Gráfica ----------
class Espectro(Widget):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.mic = None
        self.window = np.hanning(N)
        self.freqs = np.fft.rfftfreq(N, 1 / SR)
        self.mask = self.freqs <= FMAX
        self.suave = None
        if platform == "android":
            request_permissions([Permission.RECORD_AUDIO], self._permiso)
        else:
            self._permiso(None, [True])

    def _permiso(self, perms, grants):
        if all(grants):
            self.mic = Mic()
            Clock.schedule_interval(self.actualizar, 1 / 30)

    def actualizar(self, dt):
        x = self.mic.read()
        if len(x) < N:
            return
        x = (x - x.mean()) * self.window
        mag = np.abs(np.fft.rfft(x)) / N
        db = 20 * np.log10(mag[self.mask] + 1e-9)       # escala en dB
        db = np.clip((db + 100) / 100, 0, 1)            # normalizar 0..1
        self.suave = db if self.suave is None else 0.6 * self.suave + 0.4 * db
        self.dibujar(self.suave)

    def dibujar(self, y):
        self.canvas.clear()
        x = np.linspace(self.x, self.right, len(y))
        pts = np.column_stack((x, self.y + y * self.height)).ravel().tolist()
        with self.canvas:
            Color(0.1, 0.9, 0.4)
            Line(points=pts, width=1.2)

class EspectroApp(App):
    def build(self):
        return Espectro()

    def on_stop(self):
        w = self.root
        if w.mic:
            w.mic.stop()

EspectroApp().run()