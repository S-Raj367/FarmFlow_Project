import type { NextConfig } from "next";

const nextConfig: NextConfig = {

  productionBrowserSourceMaps:false,

  eslint:{

    ignoreDuringBuilds:true,

  },

  typescript:{

    ignoreBuildErrors:true,

  },

};

export default nextConfig;
