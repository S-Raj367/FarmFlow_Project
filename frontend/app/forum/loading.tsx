import { Skeleton } from "@/components/ui/skeleton"
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"

export default function Loading() {
  return (
    <div className="container py-12">
      <div className="max-w-6xl mx-auto">
        {/* Hero skeleton */}
        <div className="w-full h-[300px] bg-gray-200 rounded-lg mb-8"></div>

        {/* Search and filters skeleton */}
        <div className="flex justify-between items-center mb-8">
          <Skeleton className="h-8 w-[150px]" />
          <div className="flex gap-2">
            <Skeleton className="h-10 w-[200px]" />
            <Skeleton className="h-10 w-[120px]" />
          </div>
        </div>

        {/* Categories skeleton */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          {[...Array(6)].map((_, i) => (
            <Card key={i} className="overflow-hidden">
              <Skeleton className="h-2 w-full" />
              <CardHeader className="pb-2">
                <div className="flex items-center gap-3">
                  <Skeleton className="h-10 w-10 rounded-full" />
                  <Skeleton className="h-6 w-[120px]" />
                </div>
                <Skeleton className="h-4 w-full mt-2" />
                <Skeleton className="h-4 w-3/4 mt-1" />
              </CardHeader>
              <CardFooter className="pt-2">
                <div className="flex justify-between w-full">
                  <Skeleton className="h-4 w-[80px]" />
                  <Skeleton className="h-4 w-[80px]" />
                </div>
              </CardFooter>
            </Card>
          ))}
        </div>

        {/* Tabs skeleton */}
        <Skeleton className="h-10 w-full mb-6" />

        {/* Topics skeleton */}
        <Card>
          <CardContent className="p-0">
            <div className="divide-y">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="p-4">
                  <div className="flex items-start gap-4">
                    <Skeleton className="h-10 w-10 rounded-full hidden sm:block" />
                    <div className="flex-grow">
                      <div className="flex items-center gap-2 mb-1">
                        <Skeleton className="h-5 w-16 rounded-full" />
                        <Skeleton className="h-5 w-24 rounded-full" />
                      </div>
                      <Skeleton className="h-6 w-full" />
                      <div className="flex items-center gap-4 mt-2">
                        <Skeleton className="h-4 w-12" />
                        <Skeleton className="h-4 w-12" />
                        <Skeleton className="h-4 w-12" />
                      </div>
                    </div>
                    <div className="text-right">
                      <Skeleton className="h-4 w-20" />
                      <Skeleton className="h-4 w-16 mt-1" />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
          <CardFooter className="flex justify-center p-4">
            <Skeleton className="h-10 w-[120px]" />
          </CardFooter>
        </Card>
      </div>
    </div>
  )
}

