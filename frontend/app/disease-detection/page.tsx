"use client"

import type React from "react"
import { useState, useRef } from "react"
import Image from "next/image"
import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { AlertCircle, Upload, X } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"

export default function DiseaseDetection() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<{
    disease: string
    confidence: number
    description: string
    treatment: string
  } | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setFileName(file.name)
      const reader = new FileReader()
      reader.onload = () => {
        setSelectedImage(reader.result as string)
      }
      reader.readAsDataURL(file)
      setResult(null)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file) {
      setFileName(file.name)
      const reader = new FileReader()
      reader.onload = () => {
        setSelectedImage(reader.result as string)
      }
      reader.readAsDataURL(file)
      setResult(null)
    }
  }

  const handleRemoveImage = () => {
    setSelectedImage(null)
    setFileName(null)
    setResult(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const handleDetectDisease = async () => {
    if (!selectedImage || !fileName) return

    setLoading(true)
    setProgress(0)

    // Convert the base64 image to a file
    const file = await fetch(selectedImage).then((res) => res.blob())

    // Create a FormData object
    const formData = new FormData()
    formData.append("file", file, fileName)

    try {
      // Send the image to the Flask backend
      const response = await fetch(`${process.env.NEXT_PUBLIC_ML_SERVER_URL}/api/predict-disease`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Failed to detect disease")
      }

      const data = await response.json()
      console.log("API Response:", data)

      // Parse the response and update the result state
      const [disease, confidence] = data.disease.split(" :")
      setResult({
        disease,
        confidence: parseFloat(data.confidence),
        description: "Description not available", // Add description from backend if available
        treatment: "Treatment not available", // Add treatment from backend if available
      })
    } catch (error) {
      console.error("Error detecting disease:", error)
      alert("Failed to detect disease. Please try again.")
    } finally {
      setLoading(false)
      setProgress(100)
    }
  }

  return (
    <div className="container py-12">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 mb-2">Plant Disease Detection</h1>
          <p className="text-gray-600">
            Upload an image of your plant to identify diseases and get treatment recommendations
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Upload Plant Image</CardTitle>
            <CardDescription>Upload a clear image of the affected plant part for accurate detection</CardDescription>
          </CardHeader>
          <CardContent>
            <div
              className="border-2 border-dashed rounded-lg p-6 text-center cursor-pointer hover:bg-gray-50 transition-colors"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              {!selectedImage ? (
                <div className="space-y-4">
                  <div className="mx-auto w-12 h-12 rounded-full bg-lime-50 flex items-center justify-center">
                    <Upload className="h-6 w-6 text-lime-700" />
                  </div>
                  <div>
                    <p className="text-sm font-medium">Drag and drop your image here or click to browse</p>
                    <p className="text-xs text-gray-500 mt-1">Supports JPG, PNG, JPEG (Max 5MB)</p>
                  </div>
                </div>
              ) : (
                <div className="relative">
                  <Button
                    variant="outline"
                    size="icon"
                    className="absolute top-2 right-2 z-10 bg-white rounded-full"
                    onClick={(e) => {
                      e.stopPropagation()
                      handleRemoveImage()
                    }}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                  <div className="relative h-[300px] w-full">
                    <Image
                      src={selectedImage || "/placeholder.svg"}
                      alt="Selected plant"
                      fill
                      className="object-contain"
                    />
                  </div>
                  <p className="mt-2 text-sm text-gray-500">{fileName}</p>
                </div>
              )}
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept="image/png, image/jpeg, image/jpg"
                onChange={handleFileChange}
              />
            </div>

            {selectedImage && (
              <div className="mt-6">
                <Button
                  onClick={handleDetectDisease}
                  className="w-full bg-lime-700 hover:bg-lime-800"
                  disabled={loading}
                >
                  {loading ? "Analyzing..." : "Detect Disease"}
                </Button>

                {loading && (
                  <div className="mt-4 space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Analyzing image</span>
                      <span>{progress}%</span>
                    </div>
                    <Progress value={progress} className="h-2" />
                  </div>
                )}
              </div>
            )}

            {result && (
              <div className="mt-8 space-y-4">
                <div className="bg-lime-50 border border-lime-200 rounded-lg p-4">
                  <div className="flex justify-between items-start">
                    <h3 className="font-semibold text-lg text-lime-800">{result.disease}</h3>
                    <div className="bg-lime-700 text-white text-xs font-medium px-2.5 py-1 rounded">
                      {result.confidence.toFixed(1)}% confidence
                    </div>
                  </div>
                  <p className="mt-2 text-sm text-gray-700">{result.description}</p>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Recommended Treatment:</h4>
                  <p className="text-sm text-gray-700">{result.treatment}</p>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Additional Resources:</h4>
                  <ul className="space-y-2">
                    <li>
                      <Link href="#" className="text-sm text-lime-700 hover:underline inline-flex items-center">
                        Learn more about {result.disease}
                      </Link>
                    </li>
                    <li>
                      <Link href="#" className="text-sm text-lime-700 hover:underline inline-flex items-center">
                        Prevention strategies for {result.disease}
                      </Link>
                    </li>
                    <li>
                      <Link href="#" className="text-sm text-lime-700 hover:underline inline-flex items-center">
                        Find products to treat {result.disease}
                      </Link>
                    </li>
                  </ul>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="mt-8">
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Important Note</AlertTitle>
            <AlertDescription>
              This tool provides preliminary disease identification. For critical cases, consult with a professional
              agricultural extension service or plant pathologist.
            </AlertDescription>
          </Alert>
        </div>
      </div>
    </div>
  )
}
