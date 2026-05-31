"use client"
import { useState, useRef } from "react";
import Image from "next/image";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertCircle, Upload, X, Ruler } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function HeightEstimation() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<{
    height: number;
    growthStage: string;
    estimatedYield: string;
    nextAction: string;
  } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = () => {
        setSelectedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
      setResult(null);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = () => {
        setSelectedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
      setResult(null);
    }
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
    setFileName(null);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleEstimateHeight = async () => {
    if (!selectedImage || !fileName) return;

    setLoading(true);
    setProgress(0);

    const formData = new FormData();
    const file = fileInputRef.current?.files?.[0];
    if (file) {
      formData.append("file", file);

      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_ML_SERVER_URL}/api/predict-height', {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error("Failed to fetch height data");
        }

        const data = await response.json();
        setResult({
          height: parseFloat(data.height),
          growthStage: "Vegetative", // You can adjust this based on your logic
          estimatedYield: "Moderate", // You can adjust this based on your logic
          nextAction: "Apply fertilizer", // You can adjust this based on your logic
        });
      } catch (error) {
        console.error("Error:", error);
      } finally {
        setLoading(false);
        setProgress(100);
      }
    }
  };

  return (
    <div className="container py-12">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 mb-2">Plant Height Estimation</h1>
          <p className="text-gray-600">Upload an image of your plant to estimate its height and growth stage</p>
        </div>

        <Tabs defaultValue="upload" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-8">
            <TabsTrigger value="upload">Upload Image</TabsTrigger>
            <TabsTrigger value="instructions">Instructions</TabsTrigger>
          </TabsList>

          <TabsContent value="upload">
            <Card>
              <CardHeader>
                <CardTitle>Upload Plant Image</CardTitle>
                <CardDescription>
                  Upload a clear image of your plant with a reference object for accurate height estimation
                </CardDescription>
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
                          e.stopPropagation();
                          handleRemoveImage();
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
                      onClick={handleEstimateHeight}
                      className="w-full bg-lime-700 hover:bg-lime-800"
                      disabled={loading}
                    >
                      {loading ? "Processing..." : "Estimate Height"}
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
                  <div className="mt-8">
                    <Card className="bg-lime-50 border-lime-200">
                      <CardHeader className="pb-2">
                        <div className="flex justify-between items-center">
                          <CardTitle className="text-lime-800">Height Analysis</CardTitle>
                          <Ruler className="h-5 w-5 text-lime-700" />
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-1">
                            <p className="text-sm text-gray-500">Estimated Height</p>
                            <p className="text-2xl font-bold text-lime-700">{result.height} cm</p>
                          </div>
                          <div className="space-y-1">
                            <p className="text-sm text-gray-500">Growth Stage</p>
                            <p className="text-lg font-semibold text-lime-700">{result.growthStage}</p>
                          </div>
                          <div className="space-y-1">
                            <p className="text-sm text-gray-500">Estimated Yield</p>
                            <p className="text-lg font-semibold text-lime-700">{result.estimatedYield}</p>
                          </div>
                          <div className="space-y-1">
                            <p className="text-sm text-gray-500">Recommended Action</p>
                            <p className="text-lg font-semibold text-lime-700">{result.nextAction}</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <div className="mt-4">
                      <h4 className="font-medium mb-2">Growth Trajectory</h4>
                      <div className="h-[200px] bg-gray-100 rounded-lg flex items-center justify-center">
                        <p className="text-gray-500 text-sm">Growth chart visualization would appear here</p>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="instructions">
            <Card>
              <CardHeader>
                <CardTitle>How to Take the Perfect Plant Photo</CardTitle>
                <CardDescription>Follow these guidelines to get the most accurate height estimation</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <h3 className="font-semibold">1. Include a Reference Object</h3>
                  <p className="text-sm text-gray-600">
                    Place a standard-sized object next to your plant (e.g., a ruler, coin, or credit card). This helps
                    our algorithm calculate the exact scale.
                  </p>
                </div>

                <div className="space-y-2">
                  <h3 className="font-semibold">2. Proper Lighting</h3>
                  <p className="text-sm text-gray-600">
                    Take photos in good natural light, avoiding harsh shadows or overexposure.
                  </p>
                </div>

                <div className="space-y-2">
                  <h3 className="font-semibold">3. Capture the Entire Plant</h3>
                  <p className="text-sm text-gray-600">
                    Make sure the entire plant from soil level to the top is visible in the frame.
                  </p>
                </div>

                <div className="space-y-2">
                  <h3 className="font-semibold">4. Take Photos at Eye Level</h3>
                  <p className="text-sm text-gray-600">
                    For the most accurate measurements, take photos at the same height as the middle of the plant.
                  </p>
                </div>

                <div className="mt-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-100 rounded-lg p-4 flex items-center justify-center h-[150px]">
                      <p className="text-gray-500 text-sm text-center">Example of good photo</p>
                    </div>
                    <div className="bg-gray-100 rounded-lg p-4 flex items-center justify-center h-[150px]">
                      <p className="text-gray-500 text-sm text-center">Example of bad photo</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <div className="mt-8">
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Important Note</AlertTitle>
            <AlertDescription>
              For the most accurate results, include a reference object of known size in your photo. Height estimates
              are approximate and may vary based on image quality and perspective.
            </AlertDescription>
          </Alert>
        </div>
      </div>
    </div>
  );
}
