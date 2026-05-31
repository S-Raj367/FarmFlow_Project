"use client"
import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Leaf, AlertCircle } from 'lucide-react';

interface CropRecommendationInput {
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  temperature: number;
  humidity: number;
  ph: number;
  rainfall: number;
}

const CropRecommendation: React.FC = () => {
  const [formData, setFormData] = useState<CropRecommendationInput>({
    nitrogen: 50,
    phosphorus: 50,
    potassium: 50,
    temperature: 25,
    humidity: 50,
    ph: 7,
    rainfall: 100,
  });

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<string>('');
  const [error, setError] = useState<string>('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: parseFloat(value),
    });
  };

  const handleSliderChange = (name: string, value: number[]) => {
    setFormData({
      ...formData,
      [name]: value[0],
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      // Updated URL to the Flask backend on port 5000 with new endpoint
      const response = await fetch(`${process.env.NEXT_PUBLIC_ML_SERVER_URL}/api/predict-crop`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          N: formData.nitrogen,
          P: formData.phosphorus,
          K: formData.potassium,
          temperature: formData.temperature,
          humidity: formData.humidity,
          ph: formData.ph,
          rainfall: formData.rainfall,
        }),
      });
      
      // Additional logging to debug issues
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        throw new Error(`Server error: ${response.status}. ${errorText}`);
      }
      
      // Parse the JSON response
      try {
        const data = await response.json();
        console.log('Response data:', data);
        
        if (data && data.crop) {
          setResult(data.crop);
        } else if (data && data.error) {
          throw new Error(data.error);
        } else {
          throw new Error("Received unexpected response format");
        }
      } catch (jsonError) {
        console.error('JSON parsing error:', jsonError);
        throw new Error('Invalid response from server');
      }
    } catch (error) {
      console.error('Request error:', error);
      setError(error instanceof Error ? error.message : "Failed to get recommendation");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container py-12">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 mb-2">Crop Recommendation</h1>
          <p className="text-gray-600">Enter your soil and climate data to get personalized crop recommendations</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Soil and Climate Parameters</CardTitle>
            <CardDescription>Provide accurate measurements for the best recommendations</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="nitrogen">Nitrogen (N) - {formData.nitrogen} kg/ha</Label>
                  <Slider
                    id="nitrogen"
                    min={0}
                    max={140}
                    step={1}
                    value={[formData.nitrogen]}
                    onValueChange={(value) => handleSliderChange("nitrogen", value)}
                    className="py-4"
                  />
                  <Input
                    type="number"
                    id="nitrogen"
                    name="nitrogen"
                    value={formData.nitrogen}
                    onChange={handleChange}
                    min={0}
                    max={140}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="phosphorus">Phosphorus (P) - {formData.phosphorus} kg/ha</Label>
                  <Slider
                    id="phosphorus"
                    min={0}
                    max={140}
                    step={1}
                    value={[formData.phosphorus]}
                    onValueChange={(value) => handleSliderChange("phosphorus", value)}
                    className="py-4"
                  />
                  <Input
                    type="number"
                    id="phosphorus"
                    name="phosphorus"
                    value={formData.phosphorus}
                    onChange={handleChange}
                    min={0}
                    max={140}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="potassium">Potassium (K) - {formData.potassium} kg/ha</Label>
                  <Slider
                    id="potassium"
                    min={0}
                    max={140}
                    step={1}
                    value={[formData.potassium]}
                    onValueChange={(value) => handleSliderChange("potassium", value)}
                    className="py-4"
                  />
                  <Input
                    type="number"
                    id="potassium"
                    name="potassium"
                    value={formData.potassium}
                    onChange={handleChange}
                    min={0}
                    max={140}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="temperature">Temperature - {formData.temperature}°C</Label>
                  <Slider
                    id="temperature"
                    min={0}
                    max={50}
                    step={0.1}
                    value={[formData.temperature]}
                    onValueChange={(value) => handleSliderChange("temperature", value)}
                    className="py-4"
                  />
                  <Input
                    type="number"
                    id="temperature"
                    name="temperature"
                    value={formData.temperature}
                    onChange={handleChange}
                    min={0}
                    max={50}
                    step={0.1}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="humidity">Humidity - {formData.humidity}%</Label>
                  <Slider
                    id="humidity"
                    min={0}
                    max={100}
                    step={1}
                    value={[formData.humidity]}
                    onValueChange={(value) => handleSliderChange("humidity", value)}
                    className="py-4"
                  />
                  <Input
                    type="number"
                    id="humidity"
                    name="humidity"
                    value={formData.humidity}
                    onChange={handleChange}
                    min={0}
                    max={100}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="ph">pH - {formData.ph}</Label>
                  <Slider
                    id="ph"
                    min={0}
                    max={14}
                    step={0.1}
                    value={[formData.ph]}
                    onValueChange={(value) => handleSliderChange("ph", value)}
                    className="py-4"
                  />
                  <Input
                    type="number"
                    id="ph"
                    name="ph"
                    value={formData.ph}
                    onChange={handleChange}
                    min={0}
                    max={14}
                    step={0.1}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="rainfall">Rainfall - {formData.rainfall} mm</Label>
                  <Slider
                    id="rainfall"
                    min={0}
                    max={300}
                    step={1}
                    value={[formData.rainfall]}
                    onValueChange={(value) => handleSliderChange("rainfall", value)}
                    className="py-4"
                  />
                  <Input
                    type="number"
                    id="rainfall"
                    name="rainfall"
                    value={formData.rainfall}
                    onChange={handleChange}
                    min={0}
                    max={300}
                  />
                </div>
              </div>

              <Button type="submit" className="w-full bg-lime-700 hover:bg-lime-800" disabled={loading}>
                {loading ? "Analyzing..." : "Recommend Crop"}
              </Button>
            </form>

            {error && (
              <div className="mt-8">
                <Alert variant="destructive">
                  <AlertCircle className="h-5 w-5" />
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              </div>
            )}

            {result && !error && (
              <div className="mt-8">
                <Alert className="bg-lime-50 border-lime-200">
                  <Leaf className="h-5 w-5 text-lime-700" />
                  <AlertTitle className="text-lime-700">Recommendation</AlertTitle>
                  <AlertDescription>
                    <p className="mb-2">Based on your soil and climate data, we recommend growing:</p>
                    <p className="text-2xl font-bold text-lime-700 capitalize">{result}</p>
                  </AlertDescription>
                </Alert>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="mt-8">
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Important Note</AlertTitle>
            <AlertDescription>
              These recommendations are based on machine learning models and should be used as a guide. Always consult
              with local agricultural experts for the best results in your specific region.
            </AlertDescription>
          </Alert>
        </div>
      </div>
    </div>
  );
};

export default CropRecommendation;
