import Link from "next/link"
import { ArrowRight, Leaf, Droplets, Sprout, TrendingUp, MessageSquare, Users } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import HeroBackground from "@/components/three-js/hero-background"
import PlantGrowthVisualization from "@/components/three-js/plant-growth-visualization"

export default function Home() {
  return (
    <div>
      {/* Hero Section with 3D Background */}
      <section className="relative bg-gradient-to-b from-lime-50 to-white py-24 md:py-32 overflow-hidden">
        <HeroBackground />
        <div className="container relative z-10">
          <div className="max-w-2xl">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight text-gray-900 mb-6">
              Smart Agriculture for a <span className="text-lime-700">Sustainable Future</span>
            </h1>
            <p className="text-lg text-gray-600 max-w-md mb-8">
              FarmFlow helps farmers make data-driven decisions with AI-powered crop recommendations, disease detection,
              height estimation, and price prediction.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Button
                asChild
                size="lg"
                className="bg-lime-700 hover:bg-lime-800 transition-all duration-200 transform hover:translate-y-[-2px]"
              >
                <Link href="/crop-recommendation">Get Started</Link>
              </Button>
              <Button
                asChild
                variant="outline"
                size="lg"
                className="border-lime-200 hover:bg-lime-50 hover:text-lime-700 hover:border-lime-300 transition-all duration-200"
              >
                <Link href="/about">Learn More</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 md:py-24">
        <div className="container">
          <div className="text-center max-w-2xl mx-auto mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4 text-gray-900">Our Features</h2>
            <p className="text-lg text-gray-600">
              Comprehensive tools to optimize your farming operations and increase yield
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="border-lime-100 hover:border-lime-300 transition-all duration-300 hover:shadow-md group">
              <CardHeader className="pb-2">
                <div className="w-12 h-12 rounded-full bg-lime-100 flex items-center justify-center mb-2 group-hover:bg-lime-200 transition-colors">
                  <Leaf className="h-6 w-6 text-lime-700" />
                </div>
                <CardTitle>Crop Recommendation</CardTitle>
                <CardDescription>Get AI-powered crop suggestions based on soil and climate data</CardDescription>
              </CardHeader>
              <CardContent>
                <Link
                  href="/crop-recommendation"
                  className="text-lime-700 font-medium inline-flex items-center hover:underline"
                >
                  Try it now <ArrowRight className="ml-1 h-4 w-4" />
                </Link>
              </CardContent>
            </Card>

            <Card className="border-lime-100 hover:border-lime-300 transition-all duration-300 hover:shadow-md group">
              <CardHeader className="pb-2">
                <div className="w-12 h-12 rounded-full bg-lime-100 flex items-center justify-center mb-2 group-hover:bg-lime-200 transition-colors">
                  <Droplets className="h-6 w-6 text-lime-700" />
                </div>
                <CardTitle>Disease Detection</CardTitle>
                <CardDescription>Identify plant diseases early with image recognition technology</CardDescription>
              </CardHeader>
              <CardContent>
                <Link
                  href="/disease-detection"
                  className="text-lime-700 font-medium inline-flex items-center hover:underline"
                >
                  Try it now <ArrowRight className="ml-1 h-4 w-4" />
                </Link>
              </CardContent>
            </Card>

            <Card className="border-lime-100 hover:border-lime-300 transition-all duration-300 hover:shadow-md group">
              <CardHeader className="pb-2">
                <div className="w-12 h-12 rounded-full bg-lime-100 flex items-center justify-center mb-2 group-hover:bg-lime-200 transition-colors">
                  <Sprout className="h-6 w-6 text-lime-700" />
                </div>
                <CardTitle>Height Estimation</CardTitle>
                <CardDescription>Monitor plant growth with accurate height measurements</CardDescription>
              </CardHeader>
              <CardContent>
                <Link
                  href="/height-estimation"
                  className="text-lime-700 font-medium inline-flex items-center hover:underline"
                >
                  Try it now <ArrowRight className="ml-1 h-4 w-4" />
                </Link>
              </CardContent>
            </Card>

            <Card className="border-lime-100 hover:border-lime-300 transition-all duration-300 hover:shadow-md group">
              <CardHeader className="pb-2">
                <div className="w-12 h-12 rounded-full bg-lime-100 flex items-center justify-center mb-2 group-hover:bg-lime-200 transition-colors">
                  <TrendingUp className="h-6 w-6 text-lime-700" />
                </div>
                <CardTitle>Price Prediction</CardTitle>
                <CardDescription>Forecast crop prices to optimize selling decisions</CardDescription>
              </CardHeader>
              <CardContent>
                <Link
                  href="/price-prediction"
                  className="text-lime-700 font-medium inline-flex items-center hover:underline"
                >
                  Try it now <ArrowRight className="ml-1 h-4 w-4" />
                </Link>
              </CardContent>
            </Card>

            <Card className="border-lime-100 hover:border-lime-300 transition-all duration-300 hover:shadow-md group">
              <CardHeader className="pb-2">
                <div className="w-12 h-12 rounded-full bg-lime-100 flex items-center justify-center mb-2 group-hover:bg-lime-200 transition-colors">
                  <MessageSquare className="h-6 w-6 text-lime-700" />
                </div>
                <CardTitle>FarmBot Chat</CardTitle>
                <CardDescription>Get instant farming advice from our AI assistant</CardDescription>
              </CardHeader>
              <CardContent>
                <Link href="/chatbot" className="text-lime-700 font-medium inline-flex items-center hover:underline">
                  Try it now <ArrowRight className="ml-1 h-4 w-4" />
                </Link>
              </CardContent>
            </Card>

            <Card className="border-lime-100 hover:border-lime-300 transition-all duration-300 hover:shadow-md group">
              <CardHeader className="pb-2">
                <div className="w-12 h-12 rounded-full bg-lime-100 flex items-center justify-center mb-2 group-hover:bg-lime-200 transition-colors">
                  <Users className="h-6 w-6 text-lime-700" />
                </div>
                <CardTitle>Community Forum</CardTitle>
                <CardDescription>Connect with farmers and share knowledge and experiences</CardDescription>
              </CardHeader>
              <CardContent>
                <Link href="/forum" className="text-lime-700 font-medium inline-flex items-center hover:underline">
                  Join now <ArrowRight className="ml-1 h-4 w-4" />
                </Link>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* 3D Plant Growth Visualization Section */}
      <section className="py-16 bg-gray-50">
        <div className="container">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl font-bold mb-4 text-gray-900">Interactive Plant Growth Visualization</h2>
              <p className="text-lg text-gray-600 mb-6">
                Watch your plants grow in real-time with our advanced 3D visualization technology. Monitor growth stages
                and predict harvest times with precision.
              </p>
              <ul className="space-y-3 mb-6">
                <li className="flex items-start gap-2">
                  <div className="w-6 h-6 rounded-full bg-lime-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Leaf className="h-3 w-3 text-lime-700" />
                  </div>
                  <span className="text-gray-700">Track growth stages from seedling to maturity</span>
                </li>
                <li className="flex items-start gap-2">
                  <div className="w-6 h-6 rounded-full bg-lime-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Leaf className="h-3 w-3 text-lime-700" />
                  </div>
                  <span className="text-gray-700">Compare actual growth with expected patterns</span>
                </li>
                <li className="flex items-start gap-2">
                  <div className="w-6 h-6 rounded-full bg-lime-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Leaf className="h-3 w-3 text-lime-700" />
                  </div>
                  <span className="text-gray-700">Identify optimal harvest times for maximum yield</span>
                </li>
              </ul>
              <Button
                asChild
                className="bg-lime-700 hover:bg-lime-800 transition-all duration-200 transform hover:translate-y-[-2px]"
              >
                <Link href="/height-estimation">Try Height Estimation</Link>
              </Button>
            </div>
            <div>
              <PlantGrowthVisualization height={150} className="shadow-xl" />
            </div>
          </div>
        </div>
      </section>

      {/* Community Globe Section */}
      <section className="py-16 md:py-24">
        <div className="container">
          <div className="text-center max-w-2xl mx-auto mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4 text-gray-900">Global Farming Community</h2>
            <p className="text-lg text-gray-600">
              Join farmers from around the world sharing knowledge and best practices
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div className="h-[400px]">
              <CommunityGlobe className="h-full" />
            </div>
            <div>
              <h3 className="text-2xl font-bold mb-4 text-gray-900">Connect with Farmers Worldwide</h3>
              <p className="text-gray-600 mb-6">
                Our community spans across continents, bringing together diverse farming knowledge and techniques. Learn
                from experienced farmers and share your own insights.
              </p>

              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-lime-50 rounded-lg p-4 text-center">
                  <p className="text-2xl font-bold text-lime-800">12,450+</p>
                  <p className="text-sm text-lime-700">Active Members</p>
                </div>
                <div className="bg-lime-50 rounded-lg p-4 text-center">
                  <p className="text-2xl font-bold text-lime-800">120+</p>
                  <p className="text-sm text-lime-700">Countries</p>
                </div>
                <div className="bg-lime-50 rounded-lg p-4 text-center">
                  <p className="text-2xl font-bold text-lime-800">45,230+</p>
                  <p className="text-sm text-lime-700">Forum Posts</p>
                </div>
                <div className="bg-lime-50 rounded-lg p-4 text-center">
                  <p className="text-2xl font-bold text-lime-800">3,120+</p>
                  <p className="text-sm text-lime-700">Topics Discussed</p>
                </div>
              </div>

              <Button
                asChild
                className="bg-lime-700 hover:bg-lime-800 transition-all duration-200 transform hover:translate-y-[-2px]"
              >
                <Link href="/forum">Join Our Community</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-gradient-to-r from-lime-700 to-lime-900 py-16 relative overflow-hidden">
        <div className="absolute inset-0 bg-[url('/placeholder.svg?height=400&width=1200')] bg-cover bg-center mix-blend-overlay opacity-20"></div>
        <div className="container relative z-10 text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-white">Ready to Transform Your Farming?</h2>
          <p className="text-lg text-lime-100 max-w-2xl mx-auto mb-8">
            Join thousands of farmers who are already using FarmFlow to increase yields and reduce costs.
          </p>
          <Button
            asChild
            size="lg"
            className="bg-white text-lime-800 hover:bg-lime-100 transition-all duration-200 transform hover:translate-y-[-2px]"
          >
            <Link href="/crop-recommendation">Get Started Now</Link>
          </Button>
        </div>
      </section>
    </div>
  )
}

// Import at the top of the file
import CommunityGlobe from "@/components/three-js/community-globe"

