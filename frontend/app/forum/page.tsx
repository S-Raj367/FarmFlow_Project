"use client"

import type React from "react"

import { useState } from "react"
import Link from "next/link"
import Image from "next/image"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import {
  AlertCircle,
  MessageSquare,
  PlusCircle,
  Search,
  ThumbsUp,
  Users,
  Eye,
  Filter,
  SortAsc,
  Leaf,
  Droplets,
  Sprout,
  TrendingUp,
} from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

type ForumCategory = {
  id: string
  name: string
  description: string
  icon: React.ReactNode
  topics: number
  posts: number
}

type ForumTopic = {
  id: string
  title: string
  category: string
  author: {
    name: string
    avatar: string
  }
  replies: number
  views: number
  likes: number
  lastActivity: string
  isHot?: boolean
  isPinned?: boolean
}

export default function Forum() {
  const [searchQuery, setSearchQuery] = useState("")
  const [isAuthDialogNeeded, setIsAuthDialogNeeded] = useState(false)

  const categories: ForumCategory[] = [
    {
      id: "crop-growing",
      name: "Crop Growing",
      description: "Discussions about growing various crops and best practices",
      icon: <Leaf className="h-5 w-5 text-lime-700" />,
      topics: 124,
      posts: 1453,
    },
    {
      id: "pest-management",
      name: "Pest Management",
      description: "Strategies for managing pests and diseases in crops",
      icon: <Droplets className="h-5 w-5 text-lime-700" />,
      topics: 87,
      posts: 932,
    },
    {
      id: "soil-health",
      name: "Soil Health",
      description: "Improving and maintaining soil fertility and structure",
      icon: <Sprout className="h-5 w-5 text-lime-700" />,
      topics: 65,
      posts: 721,
    },
    {
      id: "market-trends",
      name: "Market Trends",
      description: "Discussions about agricultural markets and price trends",
      icon: <TrendingUp className="h-5 w-5 text-lime-700" />,
      topics: 42,
      posts: 389,
    },
    {
      id: "farming-tech",
      name: "Farming Technology",
      description: "Modern farming technologies and innovations",
      icon: <MessageSquare className="h-5 w-5 text-lime-700" />,
      topics: 93,
      posts: 876,
    },
    {
      id: "general",
      name: "General Discussion",
      description: "General farming topics and community discussions",
      icon: <Users className="h-5 w-5 text-lime-700" />,
      topics: 156,
      posts: 2134,
    },
  ]

  const recentTopics: ForumTopic[] = [
    {
      id: "1",
      title: "Best practices for organic tomato growing in sandy soil",
      category: "Crop Growing",
      author: {
        name: "JohnFarmer",
        avatar: "/placeholder.svg?height=40&width=40",
      },
      replies: 24,
      views: 342,
      likes: 18,
      lastActivity: "2 hours ago",
      isHot: true,
    },
    {
      id: "2",
      title: "How to identify and treat early blight in potatoes?",
      category: "Pest Management",
      author: {
        name: "SarahGreen",
        avatar: "/placeholder.svg?height=40&width=40",
      },
      replies: 16,
      views: 215,
      likes: 12,
      lastActivity: "5 hours ago",
    },
    {
      id: "3",
      title: "Upcoming changes to agricultural subsidies - what you need to know",
      category: "Market Trends",
      author: {
        name: "AgPolicyExpert",
        avatar: "/placeholder.svg?height=40&width=40",
      },
      replies: 32,
      views: 567,
      likes: 45,
      lastActivity: "1 day ago",
      isPinned: true,
    },
    {
      id: "4",
      title: "Cover crops for improving soil nitrogen content",
      category: "Soil Health",
      author: {
        name: "EcoFarmer",
        avatar: "/placeholder.svg?height=40&width=40",
      },
      replies: 19,
      views: 278,
      likes: 23,
      lastActivity: "2 days ago",
    },
    {
      id: "5",
      title: "Experiences with new automated irrigation systems?",
      category: "Farming Technology",
      author: {
        name: "TechFarmer",
        avatar: "/placeholder.svg?height=40&width=40",
      },
      replies: 27,
      views: 412,
      likes: 31,
      lastActivity: "3 days ago",
      isHot: true,
    },
    {
      id: "6",
      title: "Introducing myself - new organic farmer from Oregon",
      category: "General Discussion",
      author: {
        name: "NewFarmer23",
        avatar: "/placeholder.svg?height=40&width=40",
      },
      replies: 42,
      views: 189,
      likes: 37,
      lastActivity: "4 days ago",
    },
  ]

  const popularTopics: ForumTopic[] = [
    {
      id: "7",
      title: "Complete guide to natural pest control methods",
      category: "Pest Management",
      author: {
        name: "OrganicGuru",
        avatar: "/placeholder.svg?height=40&width=40",
      },
      replies: 156,
      views: 3245,
      likes: 287,
      lastActivity: "1 week ago",
      isPinned: true,
    },
    {
      id: "8",
      title: "Climate change effects on crop yields - long term strategies",
      category: "Crop Growing",
      author: {
        name: "ClimateAware",
        avatar: "/placeholder.svg?height=40&width=40",
      },
      replies: 124,
      views: 2876,
      likes: 231,
      lastActivity: "2 weeks ago",
      isHot: true,
    },
    {
      id: "9",
      title: "The economics of transitioning to organic farming",
      category: "Market Trends",
      author: {
        name: "EcoEconomist",
        avatar: "/placeholder.svg?height=40&width=40",
      },
      replies: 98,
      views: 1987,
      likes: 176,
      lastActivity: "3 weeks ago",
    },
    {
      id: "10",
      title: "Soil microbiome: The hidden world beneath our feet",
      category: "Soil Health",
      author: {
        name: "SoilScientist",
        avatar: "/placeholder.svg?height=40&width=40",
      },
      replies: 87,
      views: 1654,
      likes: 143,
      lastActivity: "1 month ago",
    },
    {
      id: "11",
      title: "Drone technology for crop monitoring - comprehensive review",
      category: "Farming Technology",
      author: {
        name: "DroneEnthusiast",
        avatar: "/placeholder.svg?height=40&width=40",
      },
      replies: 76,
      views: 1432,
      likes: 128,
      lastActivity: "1 month ago",
    },
    {
      id: "12",
      title: "Success stories: Small farms that made it big",
      category: "General Discussion",
      author: {
        name: "SmallFarmAdvocate",
        avatar: "/placeholder.svg?height=40&width=40",
      },
      replies: 112,
      views: 2345,
      likes: 198,
      lastActivity: "2 months ago",
    },
  ]

  const handleCreateTopic = () => {
    // In a real app, check if user is logged in
    // For demo purposes, we'll just show an alert
    setIsAuthDialogNeeded(true)
  }

  const renderTopicRow = (topic: ForumTopic) => (
    <div key={topic.id} className="group">
      <Link href={`/forum/topic/${topic.id}`}>
        <div className="p-4 border-b hover:bg-gray-50 transition-colors">
          <div className="flex items-start gap-4">
            <Avatar className="hidden sm:flex h-10 w-10 flex-shrink-0">
              <AvatarImage src={topic.author.avatar} alt={topic.author.name} />
              <AvatarFallback>{topic.author.name.substring(0, 2)}</AvatarFallback>
            </Avatar>

            <div className="flex-grow min-w-0">
              <div className="flex items-center gap-2 mb-1">
                {topic.isPinned && (
                  <Badge variant="outline" className="bg-amber-50 text-amber-700 hover:bg-amber-50">
                    Pinned
                  </Badge>
                )}
                {topic.isHot && (
                  <Badge variant="outline" className="bg-red-50 text-red-700 hover:bg-red-50">
                    Hot
                  </Badge>
                )}
                <Badge variant="outline" className="bg-lime-50 text-lime-700 hover:bg-lime-50">
                  {topic.category}
                </Badge>
              </div>

              <h3 className="font-medium text-gray-900 truncate group-hover:text-lime-700 transition-colors">
                {topic.title}
              </h3>

              <div className="flex items-center gap-4 mt-2 text-sm text-gray-500">
                <span className="flex items-center gap-1">
                  <MessageSquare className="h-4 w-4" />
                  {topic.replies}
                </span>
                <span className="flex items-center gap-1">
                  <Eye className="h-4 w-4" />
                  {topic.views}
                </span>
                <span className="flex items-center gap-1">
                  <ThumbsUp className="h-4 w-4" />
                  {topic.likes}
                </span>
              </div>
            </div>

            <div className="text-right text-sm">
              <p className="text-gray-500">by {topic.author.name}</p>
              <p className="text-gray-400 mt-1">{topic.lastActivity}</p>
            </div>
          </div>
        </div>
      </Link>
    </div>
  )

  return (
    <div className="container py-12">
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-gray-900 mb-2">Community Forum</h1>
            <p className="text-gray-600">
              Connect with farmers worldwide, share knowledge, and learn from others' experiences
            </p>
          </div>

          <div className="flex gap-2 w-full md:w-auto">
            <div className="relative flex-grow md:w-64">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500" />
              <Input
                type="search"
                placeholder="Search discussions..."
                className="pl-8"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <Button className="bg-lime-700 hover:bg-lime-800 whitespace-nowrap" onClick={handleCreateTopic}>
              <PlusCircle className="h-4 w-4 mr-2" />
              New Topic
            </Button>
          </div>
        </div>

        {isAuthDialogNeeded && (
          <Alert className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Authentication Required</AlertTitle>
            <AlertDescription className="flex justify-between items-center">
              <span>You need to sign in to create a new topic or participate in discussions.</span>
              <Button variant="outline" size="sm" className="ml-2" onClick={() => setIsAuthDialogNeeded(false)}>
                Dismiss
              </Button>
            </AlertDescription>
          </Alert>
        )}

        <div className="grid md:grid-cols-3 gap-6 mb-8">
          {categories.map((category) => (
            <Card key={category.id} className="hover:border-lime-300 transition-colors">
              <CardHeader className="pb-2">
                <div className="flex items-center gap-2">
                  {category.icon}
                  <CardTitle className="text-lg">{category.name}</CardTitle>
                </div>
                <CardDescription>{category.description}</CardDescription>
              </CardHeader>
              <CardFooter className="pt-2 text-sm text-gray-500">
                <div className="flex justify-between w-full">
                  <span>{category.topics} topics</span>
                  <span>{category.posts} posts</span>
                </div>
              </CardFooter>
            </Card>
          ))}
        </div>

        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Discussions</h2>
          <div className="flex gap-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" className="h-8 gap-1">
                  <Filter className="h-4 w-4" />
                  <span className="hidden sm:inline">Filter</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem>All Categories</DropdownMenuItem>
                <DropdownMenuItem>Crop Growing</DropdownMenuItem>
                <DropdownMenuItem>Pest Management</DropdownMenuItem>
                <DropdownMenuItem>Soil Health</DropdownMenuItem>
                <DropdownMenuItem>Market Trends</DropdownMenuItem>
                <DropdownMenuItem>Farming Technology</DropdownMenuItem>
                <DropdownMenuItem>General Discussion</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" className="h-8 gap-1">
                  <SortAsc className="h-4 w-4" />
                  <span className="hidden sm:inline">Sort</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem>Newest First</DropdownMenuItem>
                <DropdownMenuItem>Oldest First</DropdownMenuItem>
                <DropdownMenuItem>Most Replies</DropdownMenuItem>
                <DropdownMenuItem>Most Views</DropdownMenuItem>
                <DropdownMenuItem>Most Likes</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        <Tabs defaultValue="recent" className="w-full">
          <TabsList className="w-full grid grid-cols-2 mb-6">
            <TabsTrigger value="recent">Recent Discussions</TabsTrigger>
            <TabsTrigger value="popular">Popular Discussions</TabsTrigger>
          </TabsList>

          <TabsContent value="recent">
            <Card>
              <CardContent className="p-0">
                <div className="divide-y">{recentTopics.map(renderTopicRow)}</div>
              </CardContent>
              <CardFooter className="flex justify-center p-4">
                <Button variant="outline">Load More</Button>
              </CardFooter>
            </Card>
          </TabsContent>

          <TabsContent value="popular">
            <Card>
              <CardContent className="p-0">
                <div className="divide-y">{popularTopics.map(renderTopicRow)}</div>
              </CardContent>
              <CardFooter className="flex justify-center p-4">
                <Button variant="outline">Load More</Button>
              </CardFooter>
            </Card>
          </TabsContent>
        </Tabs>

        <div className="mt-8 bg-lime-50 rounded-lg p-6">
          <div className="flex flex-col md:flex-row gap-6 items-center">
            <div className="md:w-1/2">
              <h3 className="text-xl font-semibold text-lime-800 mb-2">Join Our Growing Community</h3>
              <p className="text-lime-700">
                Connect with thousands of farmers worldwide, share your experiences, ask questions, and learn from
                others.
              </p>
              <div className="flex gap-4 mt-4">
                <div className="text-center">
                  <p className="text-2xl font-bold text-lime-800">12,450+</p>
                  <p className="text-sm text-lime-700">Members</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-lime-800">45,230+</p>
                  <p className="text-sm text-lime-700">Posts</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-lime-800">3,120+</p>
                  <p className="text-sm text-lime-700">Topics</p>
                </div>
              </div>
            </div>
            <div className="md:w-1/2 flex justify-center">
              <Image
                src="/placeholder.svg?height=200&width=300"
                alt="Community illustration"
                width={300}
                height={200}
                className="rounded-lg"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

