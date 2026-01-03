"use client"

import type React from "react"
import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { AlertCircle, TrendingUp } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function PricePrediction() {
  const [formData, setFormData] = useState({
    crop: "",
    month: 1,       // Default to January
    year: 2023,     // Default to current year
    rainfall: 0,    // Default to 0 mm
  })

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{
    currentPrice: number
    predictedPrice: number
    priceChange: number
    confidence: number
    trend: "up" | "down" | "stable"
    factors: string[]
  } | null>(null)

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handleSelectChange = (name: string, value: string) => {
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handleSliderChange = (name: string, value: number[]) => {
    setFormData((prev) => ({ ...prev, [name]: value[0] }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!formData.crop) return

    setLoading(true)

    try {
      // Send only the 3 required features (Month, Year, Rainfall)
      const response = await fetch("http://localhost:5000/api/predict-price", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          crop: formData.crop,
          month: formData.month,    // Month
          year: formData.year,      // Year
          rainfall: formData.rainfall, // Rainfall
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to fetch price prediction")
      }

      const data = await response.json()

      // Calculate price change and trend based on the response
      const currentPrice = Math.floor(Math.random() * 1000) + 500 // Simulate current price
      const predictedPrice = parseFloat(data.price) // Use the predicted price from the backend
      const priceChange = predictedPrice - currentPrice
      const trend = priceChange > 0 ? "up" : priceChange < 0 ? "down" : "stable"

      setResult({
        currentPrice,
        predictedPrice,
        priceChange,
        confidence: Math.floor(Math.random() * 30) + 70, // Simulate confidence
        trend,
        factors: [
          "Seasonal demand fluctuations",
          "Regional supply constraints",
          "Weather patterns affecting yield",
          "Market competition dynamics",
          "Transportation and logistics costs",
        ]
          .sort(() => 0.5 - Math.random())
          .slice(0, 3), // Simulate influencing factors
      })
    } catch (error) {
      console.error("Error:", error)
    } finally {
      setLoading(false)
    }
  }

  const crops = [
    "wheat",
    "maize",
    "moong",
    "cotton",
    "coconut",
  ]

  return (
    <div className="container py-12">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 mb-2">Crop Price Prediction</h1>
          <p className="text-gray-600">Forecast future crop prices to optimize your selling decisions</p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <Card>
            <CardHeader>
              <CardTitle>Price Prediction Inputs</CardTitle>
              <CardDescription>Enter crop details and market factors to predict future prices</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="crop">Select Crop</Label>
                  <Select value={formData.crop} onValueChange={(value) => handleSelectChange("crop", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a crop" />
                    </SelectTrigger>
                    <SelectContent>
                      {crops.map((crop) => (
                        <SelectItem key={crop} value={crop.toLowerCase()}>
                          {crop}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="month">Month</Label>
                  <input
                    type="number"
                    id="month"
                    name="month"
                    value={formData.month}
                    onChange={(e) => handleChange(e)}
                    className="w-full p-2 border rounded"
                    min={1}
                    max={12}
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="year">Year</Label>
                  <input
                    type="number"
                    id="year"
                    name="year"
                    value={formData.year}
                    onChange={(e) => handleChange(e)}
                    className="w-full p-2 border rounded"
                    min={2000}
                    max={2023}
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="rainfall">Rainfall (mm)</Label>
                  <Slider
                    id="rainfall"
                    min={0}
                    max={300}
                    step={1}
                    value={[formData.rainfall]}
                    onValueChange={(value) => handleSliderChange("rainfall", value)}
                    className="py-4"
                  />
                  <p className="text-sm text-gray-500">{formData.rainfall} mm</p>
                </div>

                <Button
                  type="submit"
                  className="w-full bg-lime-700 hover:bg-lime-800"
                  disabled={loading || !formData.crop}
                >
                  {loading ? "Analyzing Market Data..." : "Predict Price"}
                </Button>
              </form>
            </CardContent>
          </Card>

          <div className="space-y-6">
            {result ? (
              <Tabs defaultValue="prediction" className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="prediction">Prediction</TabsTrigger>
                  <TabsTrigger value="analysis">Analysis</TabsTrigger>
                </TabsList>

                <TabsContent value="prediction">
                  <Card>
                    <CardHeader className="pb-2">
                      <div className="flex justify-between items-center">
                        <CardTitle className="capitalize">{formData.crop} Price Forecast</CardTitle>
                        <TrendingUp
                          className={`h-5 w-5 ${result.trend === "up" ? "text-green-600" : result.trend === "down" ? "text-red-600" : "text-amber-600"}`}
                        />
                      </div>
                      <CardDescription>Prediction with {result.confidence}% confidence</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-6">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-1">
                            <p className="text-sm text-gray-500">Current Price</p>
                            <p className="text-2xl font-bold">${result.currentPrice.toFixed(2)}/ton</p>
                          </div>
                          <div className="space-y-1">
                            <p className="text-sm text-gray-500">Predicted Price</p>
                            <p className="text-2xl font-bold">${result.predictedPrice.toFixed(2)}/ton</p>
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          <div
                            className={`text-sm font-medium px-2.5 py-1 rounded ${
                              result.trend === "up"
                                ? "bg-green-100 text-green-800"
                                : result.trend === "down"
                                  ? "bg-red-100 text-red-800"
                                  : "bg-amber-100 text-amber-800"
                            }`}
                          >
                            {result.priceChange > 0 ? "+" : ""}
                            {result.priceChange.toFixed(2)} (
                            {((result.priceChange / result.currentPrice) * 100).toFixed(1)}%)
                          </div>
                          <p className="text-sm text-gray-500">Expected change in 3 months</p>
                        </div>

                        <div className="h-[150px] bg-gray-100 rounded-lg flex items-center justify-center">
                          <p className="text-gray-500 text-sm">Price trend chart would appear here</p>
                        </div>

                        <div>
                          <h4 className="font-medium mb-2">Selling Recommendation</h4>
                          <p className="text-sm text-gray-700">
                            {result.trend === "up"
                              ? "Consider holding your crop for potential price increases in the coming months."
                              : result.trend === "down"
                                ? "Consider selling soon to avoid potential price decreases in the coming months."
                                : "Prices are expected to remain stable. Sell based on your storage capacity and cash flow needs."}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="analysis">
                  <Card>
                    <CardHeader>
                      <CardTitle>Market Analysis</CardTitle>
                      <CardDescription>Key factors influencing the price prediction</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <h4 className="font-medium">Top Influencing Factors:</h4>
                        <ul className="space-y-2">
                          {result.factors.map((factor, index) => (
                            <li key={index} className="flex items-start gap-2">
                              <div className="h-5 w-5 rounded-full bg-lime-100 text-lime-800 flex items-center justify-center flex-shrink-0 text-xs font-medium">
                                {index + 1}
                              </div>
                              <p className="text-sm text-gray-700">{factor}</p>
                            </li>
                          ))}
                        </ul>

                        <div className="mt-6">
                          <h4 className="font-medium mb-2">Market Conditions</h4>
                          <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-1">
                              <p className="text-sm text-gray-500">Rainfall</p>
                              <div className="h-2 bg-gray-200 rounded-full">
                                <div
                                  className="h-2 bg-lime-600 rounded-full"
                                  style={{ width: `${(formData.rainfall / 300) * 100}%` }}
                                ></div>
                              </div>
                              <p className="text-xs text-gray-500">{formData.rainfall} mm</p>
                            </div>
                          </div>
                        </div>

                        <div className="mt-4">
                          <h4 className="font-medium mb-2">Historical Context</h4>
                          <p className="text-sm text-gray-700">
                            Based on historical data, {formData.crop} prices typically
                            {result.trend === "up"
                              ? " increase during this season due to higher demand and limited supply."
                              : result.trend === "down"
                                ? " decrease during this season as new harvests enter the market."
                                : " remain stable during this season with balanced supply and demand."}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            ) : (
              <Card className="h-full flex flex-col justify-center items-center p-8">
                <div className="text-center space-y-4">
                  <TrendingUp className="h-12 w-12 text-lime-700 mx-auto" />
                  <h3 className="text-xl font-semibold">Price Prediction</h3>
                  <p className="text-gray-500 text-sm">
                    Select a crop and enter market factors to see price predictions and analysis
                  </p>
                </div>
              </Card>
            )}

            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Disclaimer</AlertTitle>
              <AlertDescription>
                Price predictions are based on historical data and market trends. Actual prices may vary due to
                unforeseen market conditions, policy changes, or extreme weather events.
              </AlertDescription>
            </Alert>
          </div>
        </div>
      </div>
    </div>
  )
}