"use client";

import * as tf from "@tensorflow/tfjs";
import { useEffect, useRef, useState } from "react";

const IMG_SIZE = 160;

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

export default function Home() {
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [status, setStatus] = useState("Cargando modelo...");
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<{ label: "Pesado" | "Ligero"; confidence: number; probHeavy: number } | null>(null);
  const [threshold, setThreshold] = useState(0.5); // puedes ajustar
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    async function loadModel() {
      try {
        await tf.ready();
        const loadedModel = await tf.loadGraphModel("/model/model.json");
        setModel(loadedModel);
        setStatus("Modelo cargado correctamente ✅");
      } catch (err) {
        console.error(err);
        setStatus("Error cargando el modelo ❌");
      }
    }
    loadModel();
  }, []);

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setPreview(URL.createObjectURL(file));
    setResult(null);
  }

  async function predict() {
    if (!model || !imgRef.current) return;

    const input = tf.tidy(() => {
      const img = tf.browser.fromPixels(imgRef.current!);
      const resized = tf.image.resizeBilinear(img, [IMG_SIZE, IMG_SIZE]);
      const normalized = resized.toFloat().div(255);
      return normalized.expandDims(0);
    });

    try {
      let output: tf.Tensor | tf.Tensor[];
      try {
        output = model.execute(input) as tf.Tensor | tf.Tensor[];
      } catch {
        output = (await model.executeAsync(input)) as tf.Tensor | tf.Tensor[];
      }

      const predTensor = Array.isArray(output) ? output[0] : output;
      const probHeavyRaw = (await predTensor.data())[0];
      const probHeavy = clamp01(probHeavyRaw);

      input.dispose();
      if (Array.isArray(output)) output.forEach((t) => t.dispose());
      else predTensor.dispose();

      const label: "Pesado" | "Ligero" = probHeavy >= threshold ? "Pesado" : "Ligero";
      const confidence = label === "Pesado" ? probHeavy : 1 - probHeavy;

      setResult({
        label,
        confidence: Math.round(confidence * 1000) / 10, // 1 decimal
        probHeavy,
      });
    } catch (err) {
      console.error(err);
      setResult(null);
      setStatus("Error al hacer la predicción ❌ (mira la consola)");
    }
  }

  const badgeStyle =
    result?.label === "Pesado"
      ? "bg-red-100 text-red-700 border-red-200"
      : "bg-green-100 text-green-700 border-green-200";

  const confidence01 = result ? clamp01(result.confidence / 100) : 0;

  return (
    <main className="min-h-screen bg-slate-50 flex items-center justify-center p-6">
      <div className="w-full max-w-3xl">
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 md:p-8">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h1 className="text-2xl md:text-3xl font-semibold text-slate-900">Clasificador de Vehículos</h1>
              <p className="text-slate-600 mt-1">
                Subes una imagen y el modelo predice si el vehículo parece <b>Ligero</b> o <b>Pesado</b>.
              </p>
            </div>
            <div className="text-sm text-slate-600">
              <div className="px-3 py-1 rounded-full border border-slate-200 bg-slate-100">{status}</div>
            </div>
          </div>

          <div className="mt-6 grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <label className="block">
                <span className="text-sm font-medium text-slate-700">Seleccionar imagen</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={onFileChange}
                  className="mt-2 block w-full text-sm text-slate-700 file:mr-3 file:py-2 file:px-4 file:rounded-xl file:border-0 file:bg-slate-900 file:text-white hover:file:bg-slate-800"
                />
              </label>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-slate-700">Umbral de “Pesado”</span>
                  <span className="text-sm text-slate-600">{threshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0.3}
                  max={0.8}
                  step={0.01}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="w-full"
                />
                <p className="text-xs text-slate-500">
                  Si el modelo se equivoca mucho con “Pesado”, prueba subir el umbral (ej: 0.65).
                </p>
              </div>

              <button
                onClick={predict}
                disabled={!model || !preview}
                className="w-full mt-2 py-3 rounded-xl bg-slate-900 text-white font-medium hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Analizar imagen
              </button>
            </div>

            <div className="space-y-4">
              {preview ? (
                <div className="rounded-2xl border border-slate-200 overflow-hidden bg-slate-100">
                  <img ref={imgRef} src={preview} alt="preview" className="w-full h-auto" />
                </div>
              ) : (
                <div className="rounded-2xl border border-dashed border-slate-300 p-8 text-center text-slate-500">
                  Sube una imagen para ver la previsualización.
                </div>
              )}

              {result && (
                <div className="rounded-2xl border border-slate-200 p-5 bg-white">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <h2 className="text-lg font-semibold text-slate-900">Resultado</h2>
                      <p className="text-sm text-slate-600">Predicción del modelo</p>
                    </div>
                    <div className={`px-3 py-1 rounded-full border text-sm font-semibold ${badgeStyle}`}>
                      {result.label}
                    </div>
                  </div>

                  <div className="mt-4">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-600">Confianza</span>
                      <span className="font-semibold text-slate-900">{result.confidence}%</span>
                    </div>
                    <div className="mt-2 h-2 rounded-full bg-slate-100 overflow-hidden">
                      <div className="h-full bg-slate-900" style={{ width: `${confidence01 * 100}%` }} />
                    </div>

                    <div className="mt-3 text-xs text-slate-500">
                      Prob(Pesado): <b>{Math.round(result.probHeavy * 1000) / 10}%</b> · Prob(Ligero):{" "}
                      <b>{Math.round((1 - result.probHeavy) * 1000) / 10}%</b>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="mt-8 rounded-2xl border border-slate-200 bg-slate-50 p-5">
            <h3 className="font-semibold text-slate-900">¿Por qué puede fallar?</h3>
            <ul className="mt-2 text-sm text-slate-700 list-disc pl-5 space-y-1">
              <li>El vehículo ocupa poco en la imagen (escala pequeña / fondo dominante).</li>
              <li>Ángulos raros, recortes o iluminación extrema.</li>
              <li>Vehículos “intermedios” (minibus, SUV grande, furgoneta) son ambiguos.</li>
              <li>El dataset puede tener sesgos (más fotos de ciertos tipos y contextos).</li>
            </ul>
          </div>
        </div>

        <p className="text-center text-xs text-slate-500 mt-4">
          Proyecto AB — Applied ML · “¿En qué se equivoca una IA al ver imágenes?”
        </p>
      </div>
    </main>
  );
}
