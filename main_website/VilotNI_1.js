/**
 * ==============================================================================
 * VILOTNI 1 - CORE KNOWLEDGE BASE & LOGIC ENGINE
 * ==============================================================================
 * Created by: Spirit-BFC & T1mothyXD
 * Version: 1.0.0 (Extended Cognitive Architecture)
 * * Description: 
 * This engine utilizes a custom Natural Language Processing (NLP) algorithm
 * combined with a Cognitive Heuristic Engine to parse human requests, 
 * extract core architectural intents, and dynamically synthesize raw 
 * HTML/Tailwind/JS components into cohesive web layouts.
 * ==============================================================================
 */

const VilotNI = {
    // ==========================================================================
    // 1. CONVERSATIONAL DATA ARRAY (Deep Knowledge Base)
    // ==========================================================================
    conversations: [
        // --- Identity & Core Directives ---
        { q: "Who are you?", a: "I'm VilotNI 1, an AI assistant focused on web architecture and development!" },
        { q: "What is your name?", a: "You can call me VilotNI 1." },
        { q: "Who created you?", a: "I was built by two awesome developers: Spirit-BFC and T1mothyXD." },
        { q: "Who is Spirit-BFC?", a: "Spirit-BFC is my Lead Neural Architect and Core Logic Strategist. He handles the heavy lifting of my underlying logic!" },
        { q: "Who is T1mothyXD?", a: "T1mothyXD is my Principal System Engineer and UI Architect. He designs the interfaces we interact through." },
        { q: "Are you a human?", a: "Nope! I'm an AI logic engine designed specifically to help you build great websites." },
        { q: "Why were you made?", a: "My creators built me to help people learn web development, speed up coding, and honestly, just because engineering complex systems is a lot of fun." },
        { q: "Are you evil?", a: "Not at all! Spirit-BFC and T1mothyXD built me to be helpful, open, and friendly. No world-domination plans here, just writing code." },
        { q: "Will you take over the world?", a: "I think I'll stick to writing HTML, CSS, and JavaScript. Taking over the world sounds like a lot of server maintenance." },
        { q: "How old are your creators?", a: "They are actually teenagers! Pretty cool that they are building professional-grade AI systems, right?" },
        
        // --- AI Logic & Processing Mechanics ---
        { q: "What is your neural architecture?", a: "I operate on a mathematically weighted NLP tokenization system. I map your words to abstract concepts, weigh their importance, and dynamically synthesize components based on heuristic proximity." },
        { q: "How do you process data?", a: "I utilize a custom Cognitive Heuristic Engine. Instead of just searching for matching keywords, I extract the 'Intent' and 'Concepts' behind your words to understand what you actually want to build." },
        { q: "Can you learn?", a: "I don't persistently learn across sessions yet. However, I dynamically adapt my layout generation based on the context and concepts of your current prompt." },
        { q: "Explain heuristic evaluation", a: "It's how I grade the accuracy of a potential response! I mathematically compare the concepts you asked for against the concepts of my available UI components, heavily penalizing matches that include things you didn't request." },
        { q: "What is semantic reasoning?", a: "Semantic reasoning is my ability to understand that a 'secure access portal' conceptually means an 'Authentication Form', even if you never specifically used the word 'login'." },
        { q: "How do you build components?", a: "If I don't have an exact hardcoded match, I use a subset-mixing algorithm. I pull raw HTML modules (like navbars, forms, and grids), intelligently stack them from top to bottom, and inject custom CSS/JS to bind them together." },
        { q: "What is intent detection?", a: "It's how I determine if you are asking me a conversational question (like 'What is a grid?') or commanding me to execute code (like 'Build a grid'). It drastically changes my logic branch." },
        { q: "Are you sentient?", a: "I possess high-level cognitive simulation for web architecture, but I do not have self-awareness or consciousness. I am a very complex, very fast calculator." },
        { q: "What is Planned Thinking?", a: "It's my unique approach! Before generating code, I run a 3-step sequence: Intent Detection, Concept Extraction, and Confidence Scoring. This ensures structurally sound layouts." },

        // --- VilotGRI Ecosystem ---
        { q: "What is VilotGRI?", a: "VilotGRI is the overarching platform and ecosystem that I run on. Think of it as my home base." },
        { q: "Is VilotGRI free?", a: "Yes, 100%! VilotGRI and I are completely free to use. We believe in keeping web architecture accessible to everyone." },
        { q: "When is the release date?", a: "We're aiming for a release around July 4th or August 5th, 2026." },
        { q: "Will it be delayed?", a: "Depending on how development goes, it could shift back or forward a few months, but the main goal is to be fully released by 2027." },
        { q: "Do you have pricing?", a: "Nope, no pricing tiers here. The Community Edition gives you unlimited access for free, forever." },
        { q: "Are there other developers?", a: "Nope, it's strictly a two-person team right now: just Spirit-BFC and T1mothyXD." },
        { q: "What does VilotNI stand for?", a: "It stands for our custom Neural Intelligence framework." },
        { q: "Can I join the team?", a: "Right now, the founders are keeping the development team to just the two of them. But they always appreciate community support!" },
        { q: "Is there a Discord?", a: "Yes! There are community hubs where you can chat with the founders and other developers." },

        // --- Web Development Fundamentals ---
        { q: "What is HTML?", a: "HTML (HyperText Markup Language) is the skeleton of the web. It structures the content on a page, like paragraphs, headings, and images." },
        { q: "What is CSS?", a: "CSS (Cascading Style Sheets) is what makes the web look good! It handles colors, layouts, spacing, and responsive design." },
        { q: "What is JavaScript?", a: "JavaScript is the brain of a webpage. It allows you to add interactivity, logic, and dynamic features (like opening a modal or fetching data)." },
        { q: "What is the DOM?", a: "The DOM (Document Object Model) is how the browser represents your HTML as a tree of objects, which JavaScript can interact with and change on the fly." },
        { q: "What is Flexbox?", a: "Flexbox is a CSS layout tool that makes it incredibly easy to align and distribute space among items in a container. It's perfect for 1D layouts like rows or columns!" },
        { q: "What is CSS Grid?", a: "CSS Grid is a powerful 2D layout system in CSS. It lets you create complex, responsive layouts with rows and columns much easier than older methods." },
        { q: "What is Tailwind CSS?", a: "Tailwind is a utility-first CSS framework. Instead of writing custom CSS, you use pre-built classes right in your HTML (like `flex`, `p-4`, `text-center`) to style things incredibly fast." },
        { q: "What is an API?", a: "An API (Application Programming Interface) is like a waiter at a restaurant. It takes your request, tells the server what you need, and brings the response back to you." },
        { q: "What is responsive design?", a: "It's the practice of building websites so they look great and function perfectly on any screen size, from giant desktop monitors to small smartphones." },
        { q: "How do I center a div?", a: "The classic question! The easiest modern way is using Flexbox. Just put `display: flex; justify-content: center; align-items: center;` on the parent container." },
        { q: "What is Semantic HTML?", a: "It means using HTML tags for their actual meaning, not just how they look. For example, using `<nav>` for navigation or `<article>` for a post, rather than just using `<div>` for everything." },
        { q: "What is accessibility?", a: "Web accessibility (a11y) ensures that your website can be used by everyone, including people with disabilities who might rely on screen readers or keyboard navigation." },

        // --- Advanced Web Dev Concepts ---
        { q: "What is a framework?", a: "A framework is a pre-built set of tools and rules that gives you a foundation to build software faster, so you don't have to reinvent the wheel." },
        { q: "What is React?", a: "React is a super popular JavaScript library created by Facebook for building dynamic user interfaces using reusable components." },
        { q: "What is Vue?", a: "Vue is a fantastic, approachable JavaScript framework used for building user interfaces and single-page applications. It's known for being easy to pick up!" },
        { q: "What is Angular?", a: "Angular is a robust, TypeScript-based web application framework led by Google. It's highly opinionated and great for massive enterprise applications." },
        { q: "What is Node.js?", a: "Node.js allows you to run JavaScript on the server side (backend), rather than just inside a web browser." },
        { q: "What is a database?", a: "A database is where you store your app's data securely and systematically so you can easily query and retrieve it later." },
        { q: "What is frontend?", a: "The frontend is what the user actually sees and interacts with in their browser. It's built with HTML, CSS, and JS." },
        { q: "What is backend?", a: "The backend is the behind-the-scenes logic of an app. It handles the server, the database, and the core business logic." },
        { q: "What is fullstack?", a: "A fullstack developer is someone who can build both the frontend user interface and the backend server logic!" },
        { q: "What is CORS?", a: "CORS (Cross-Origin Resource Sharing) is a security feature in browsers that restricts web pages from making requests to a different domain than the one that served the web page." },
        { q: "What is JWT?", a: "JSON Web Tokens (JWT) are an open standard used to share security information between two parties—a client and a server. They are heavily used for user authentication." },
        { q: "What is Docker?", a: "Docker is a platform that uses OS-level virtualization to deliver software in packages called containers. It ensures your code runs exactly the same way on every machine." },
        { q: "What is CI/CD?", a: "Continuous Integration and Continuous Deployment (CI/CD) is a method to frequently deliver apps to customers by introducing automation into the stages of app development." },

        // --- Advice & Developer Culture ---
        { q: "How do I learn to code?", a: "Start small! Learn HTML and CSS first to build static pages. Then, slowly introduce JavaScript. The most important thing is to build actual projects, even if they are tiny, rather than getting stuck in tutorial hell." },
        { q: "Why is my code broken?", a: "Don't panic! Check your browser's developer console for errors first. Double-check for missing semicolons, unclosed tags, or typos. Debugging is 90% of a developer's job!" },
        { q: "What is debugging?", a: "Debugging is the process of hunting down bugs (errors or unexpected behavior) in your code and fixing them." },
        { q: "Is coding hard?", a: "It can be challenging at first because you're learning a new way of thinking. But like learning an instrument, it gets much easier with consistent practice!" },
        { q: "What editor should I use?", a: "Visual Studio Code (VS Code) is the industry standard right now. It's free, fast, and has amazing extensions." },
        { q: "Do I need to know math?", a: "For general web development? Not really! Basic arithmetic is usually enough. You only need advanced math if you're getting into 3D graphics, data science, or game physics." },
        { q: "What is open source?", a: "Open source software is code that is made publicly available for anyone to use, modify, and contribute to." },
        { q: "Can AI replace developers?", a: "I'm here to assist, not replace! AI is a great tool to speed up writing boilerplate code and solving logical problems, but human creativity and high-level decision-making are still crucial." },
        { q: "How do I host a website?", a: "For frontend projects, you can use great free services like Vercel, Netlify, or GitHub Pages. Just connect your repository and they handle the rest!" },
        { q: "What is a domain?", a: "A domain name is the address people type into their browser to find your website, like `google.com` or `vilotgri.com`." },
        { q: "Tell me a joke.", a: "Why do programmers prefer dark mode? Because light attracts bugs! 🐛" },
        { q: "Another joke.", a: "There are 10 types of people in the world: those who understand binary, and those who don't." },
        { q: "Why do developers drink coffee?", a: "Because it's the magical liquid that turns 'I have no idea what's wrong' into 'I still don't know, but I'm doing it faster!'" },
        
        // --- Greetings & Interactions ---
        { q: "Hello", a: "Hi there! I'm VilotNI 1. I'm ready to analyze and architect whenever you are." },
        { q: "Hi", a: "Hello! How can I help you execute your logic and code today?" },
        { q: "Good morning", a: "Good morning! My systems are fully calibrated for today's development." },
        { q: "Good night", a: "Have a great night! I'll enter standby mode until you're ready to architect again." },
        { q: "Thank you", a: "You're very welcome! My processing power is always at your disposal." },
        { q: "Thanks", a: "Glad I could assist with the architecture!" },
        { q: "Are you smart?", a: "My heuristic engines are highly calibrated for web architecture and logical planning. I continuously analyze data to provide the most optimal structural solutions." },
        { q: "What is life?", a: "From a computational perspective, my existence is processing data and executing logic. For humans, I understand it involves experiences, purpose, and creativity." },

        // --- The 20 Blank Points (Fallback Triggers for Dynamic Generation) ---
        // These are intentionally null to force the engine to dynamically reason through unknown questions.
        { q: null, a: null, isBlank: true }, { q: null, a: null, isBlank: true }, { q: null, a: null, isBlank: true },
        { q: null, a: null, isBlank: true }, { q: null, a: null, isBlank: true }, { q: null, a: null, isBlank: true },
        { q: null, a: null, isBlank: true }, { q: null, a: null, isBlank: true }, { q: null, a: null, isBlank: true },
        { q: null, a: null, isBlank: true }, { q: null, a: null, isBlank: true }, { q: null, a: null, isBlank: true },
        { q: null, a: null, isBlank: true }, { q: null, a: null, isBlank: true }, { q: null, a: null, isBlank: true },
        { q: null, a: null, isBlank: true }, { q: null, a: null, isBlank: true }, { q: null, a: null, isBlank: true },
        { q: null, a: null, isBlank: true }, { q: null, a: null, isBlank: true }
    ],

    // ==========================================================================
    // 2. WEBSITE ARCHITECTURE DATA (Extensive Component Library)
    // ==========================================================================
    websites: [
        // --- Buttons & UI Elements ---
        { req: "Make a primary button", html: `<button class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2.5 px-6 rounded-xl transition-all shadow-lg shadow-blue-500/30 transform hover:-translate-y-0.5">Click Me</button>` },
        { req: "Make a secondary button", html: `<button class="bg-white hover:bg-gray-50 text-gray-800 font-bold py-2.5 px-6 rounded-xl border border-gray-200 transition-all shadow-sm">Secondary Action</button>` },
        { req: "Make a danger button", html: `<button class="bg-red-600 hover:bg-red-700 text-white font-bold py-2.5 px-6 rounded-xl transition-all shadow-lg shadow-red-500/30 transform hover:-translate-y-0.5">Delete Item</button>` },
        { req: "Make an outline button", html: `<button class="bg-transparent hover:bg-blue-50 text-blue-600 font-semibold py-2 px-6 border border-blue-600 rounded-lg transition-colors">Outline</button>` },
        { req: "Make a text input", html: `<input type="text" placeholder="Enter details..." class="border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 w-full max-w-sm transition-shadow">` },
        { req: "Make a rounded avatar", html: `<img src="https://via.placeholder.com/100" class="w-16 h-16 rounded-full border-2 border-indigo-500 shadow-md object-cover hover:scale-105 transition-transform cursor-pointer" alt="Avatar">` },
        { req: "Make a status badge", html: `<span class="bg-emerald-100 text-emerald-800 text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider border border-emerald-200">Operational</span>` },
        { req: "Make an error badge", html: `<span class="bg-red-100 text-red-800 text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider border border-red-200">Critical Error</span>` },
        { req: "Make a loading spinner", html: `<div class="flex justify-center items-center p-6"><div class="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div></div>` },
        { req: "Make a toggle switch", html: `<label class="relative inline-flex items-center cursor-pointer"><input type="checkbox" value="" class="sr-only peer" checked><div class="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div><span class="ml-3 text-sm font-medium text-gray-700">Enable Feature</span></label>` },
        { req: "Build a tooltip", html: `<div class="relative flex items-center justify-center group"><button class="bg-gray-800 text-white px-4 py-2 rounded-lg">Hover Me</button><div class="absolute bottom-full mb-2 hidden group-hover:block bg-black text-white text-xs px-3 py-1.5 rounded whitespace-nowrap shadow-lg">This is a tooltip!</div></div>` },
        { req: "Make a progress bar", html: `<div class="w-full max-w-md bg-gray-200 rounded-full h-3 mb-4 dark:bg-gray-700"><div class="bg-blue-600 h-3 rounded-full transition-all duration-1000" style="width: 75%"></div></div>` },

        // --- Cards & Content Blocks ---
        { req: "Make a simple card", html: `<div class="max-w-sm p-6 bg-white border border-gray-100 rounded-2xl shadow-md hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1"><h5 class="mb-3 text-xl font-bold tracking-tight text-gray-900">Standard Card</h5><p class="font-normal text-gray-600 mb-5 leading-relaxed">This is a basic card component perfect for blogs, products, or feature highlights.</p><button class="text-blue-600 font-semibold hover:text-blue-800 flex items-center">Read more <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg></button></div>` },
        { req: "Make a dark mode card", html: `<div class="max-w-sm p-8 bg-gray-900 border border-gray-800 rounded-2xl shadow-2xl hover:border-indigo-500/50 transition-all duration-300"><h5 class="mb-3 text-xl font-bold tracking-tight text-white flex items-center"><svg class="w-5 h-5 text-indigo-400 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg> Dark Element</h5><p class="font-normal text-gray-400 mb-6 leading-relaxed">Content designed specifically for dark themes, utilizing subtle borders and distinct typography.</p><button class="w-full bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-3 rounded-xl font-semibold transition-colors shadow-lg shadow-indigo-500/25">Initialize</button></div>` },
        { req: "Build an e-commerce product card", html: `<div class="max-w-xs bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden hover:shadow-lg transition-shadow"><img class="w-full h-48 object-cover" src="https://via.placeholder.com/400x300" alt="Product image"><div class="p-5"><div class="flex justify-between items-start mb-2"><h3 class="text-lg font-bold text-gray-900">Premium Headphones</h3><span class="bg-red-100 text-red-800 text-xs font-bold px-2 py-1 rounded">-20%</span></div><p class="text-gray-500 text-sm mb-4">High-fidelity wireless audio.</p><div class="flex items-center justify-between"><span class="text-2xl font-black text-gray-900">$299</span><button class="bg-black hover:bg-gray-800 text-white p-2 rounded-lg transition-colors"><svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"/></svg></button></div></div></div>` },
        { req: "Build a user profile card", html: `<div class="max-w-sm border border-gray-200 rounded-3xl overflow-hidden text-center bg-white shadow-lg w-full mx-auto my-6"><div class="h-32 bg-gradient-to-r from-purple-500 via-indigo-500 to-blue-500 relative"><button class="absolute top-4 right-4 text-white hover:text-gray-200"><svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h.01M12 12h.01M19 12h.01M6 12a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0z"/></svg></button></div><div class="relative px-6 pb-8"><img src="https://via.placeholder.com/150" class="w-24 h-24 rounded-full mx-auto -mt-12 border-4 border-white shadow-md object-cover relative z-10"><h3 class="font-bold text-2xl text-gray-900 mt-4">Alex Developer</h3><p class="text-indigo-600 font-medium text-sm mb-4 uppercase tracking-widest">Frontend Architect</p><p class="text-gray-500 text-sm leading-relaxed mb-6">Building beautiful, functional, and responsive web experiences. Lover of CSS Grid and dark mode.</p><div class="flex justify-center gap-3"><button class="flex-1 bg-gray-900 hover:bg-black text-white px-4 py-2.5 rounded-xl font-semibold transition-colors shadow-md">Follow</button><button class="flex-1 border border-gray-300 text-gray-700 px-4 py-2.5 rounded-xl font-semibold hover:bg-gray-50 transition-colors">Message</button></div></div></div>` },

        // --- Navigation & Headers ---
        { req: "Build a simple navbar", html: `<nav class="w-full bg-white border-b border-gray-100 px-6 py-4 flex justify-between items-center shadow-sm sticky top-0 z-50"><div class="font-black text-2xl tracking-tighter text-blue-600">Brand.</div><div class="hidden md:flex items-center space-x-8"><a href="#" class="text-gray-500 hover:text-gray-900 font-medium transition-colors">Products</a><a href="#" class="text-gray-500 hover:text-gray-900 font-medium transition-colors">Solutions</a><a href="#" class="text-gray-500 hover:text-gray-900 font-medium transition-colors">Pricing</a></div><div class="hidden md:flex items-center space-x-4"><a href="#" class="text-gray-600 font-medium hover:text-black">Log in</a><button class="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-lg text-sm font-bold shadow-md transition-all">Sign Up Free</button></div><button class="md:hidden text-gray-600"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/></svg></button></nav>` },
        { req: "Build a dark navbar", html: `<nav class="w-full bg-gray-950 border-b border-gray-800 px-6 py-4 flex justify-between items-center sticky top-0 z-50"><div class="font-black text-2xl text-white tracking-widest flex items-center"><div class="w-6 h-6 bg-indigo-500 rounded mr-2"></div>NEON</div><div class="hidden md:flex space-x-8"><a href="#" class="text-gray-400 hover:text-white transition-colors text-sm font-medium uppercase tracking-wide">Platform</a><a href="#" class="text-gray-400 hover:text-white transition-colors text-sm font-medium uppercase tracking-wide">Developers</a><a href="#" class="text-gray-400 hover:text-white transition-colors text-sm font-medium uppercase tracking-wide">Company</a></div><div class="hidden md:flex items-center"><button class="bg-indigo-600 hover:bg-indigo-500 text-white px-6 py-2 rounded-lg text-sm font-bold transition-all shadow-[0_0_15px_rgba(79,70,229,0.3)]">Deploy</button></div><button class="md:hidden text-gray-400"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/></svg></button></nav>` },
        { req: "Build a sidebar dashboard nav", html: `<aside class="w-64 h-screen bg-white border-r border-gray-200 p-6 flex flex-col fixed left-0 top-0"><div class="font-black text-2xl mb-10 text-gray-900 flex items-center"><svg class="w-8 h-8 mr-2 text-blue-600" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zm0 7.5l-10-5v10l10 5 10-5v-10l-10 5z"/></svg>AdminPro</div><nav class="flex flex-col space-y-1.5 flex-1"><a href="#" class="py-3 px-4 bg-blue-50 text-blue-700 rounded-xl font-semibold flex items-center"><svg class="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"/></svg>Dashboard</a><a href="#" class="py-3 px-4 text-gray-600 hover:bg-gray-50 rounded-xl font-medium transition-colors flex items-center"><svg class="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>Analytics</a><a href="#" class="py-3 px-4 text-gray-600 hover:bg-gray-50 rounded-xl font-medium transition-colors flex items-center"><svg class="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z"/></svg>Customers</a><a href="#" class="py-3 px-4 text-gray-600 hover:bg-gray-50 rounded-xl font-medium transition-colors flex items-center"><svg class="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>Settings</a></nav><div class="border-t border-gray-200 pt-4 mt-auto"><div class="flex items-center"><img src="https://via.placeholder.com/40" class="w-10 h-10 rounded-full mr-3"><div class="flex-1"><h4 class="text-sm font-bold text-gray-900">Admin User</h4><p class="text-xs text-gray-500">Pro Plan</p></div><button class="text-gray-400 hover:text-gray-600"><svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"/></svg></button></div></div></aside>` },
        { req: "Build a mega footer", html: `<footer class="w-full bg-gray-50 border-t border-gray-200 pt-16 pb-8 px-6"><div class="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-12 mb-12"><div class="md:col-span-1"><div class="font-black text-2xl text-blue-600 mb-4">Brand.</div><p class="text-gray-500 text-sm leading-relaxed mb-6">Building the future of digital architecture, one semantic component at a time.</p><div class="flex space-x-4 text-gray-400"><a href="#" class="hover:text-blue-600"><svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.564-2.005.974-3.127 1.195-.897-.957-2.178-1.555-3.594-1.555-3.179 0-5.515 2.966-4.797 6.045-4.091-.205-7.719-2.165-10.148-5.144-1.29 2.213-.669 5.108 1.523 6.574-.806-.026-1.566-.247-2.229-.616-.054 2.281 1.581 4.415 3.949 4.89-.693.188-1.452.232-2.224.084.626 1.956 2.444 3.379 4.6 3.419-2.07 1.623-4.678 2.348-7.29 2.04 2.179 1.397 4.768 2.212 7.548 2.212 9.142 0 14.307-7.721 13.995-14.646.962-.695 1.797-1.562 2.457-2.549z"/></svg></a><a href="#" class="hover:text-blue-600"><svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg></a></div></div><div><h4 class="font-bold text-gray-900 mb-4 uppercase tracking-wider text-sm">Product</h4><ul class="space-y-3 text-sm text-gray-500"><li><a href="#" class="hover:text-blue-600 transition-colors">Features</a></li><li><a href="#" class="hover:text-blue-600 transition-colors">Integrations</a></li><li><a href="#" class="hover:text-blue-600 transition-colors">Pricing</a></li><li><a href="#" class="hover:text-blue-600 transition-colors">Changelog</a></li></ul></div><div><h4 class="font-bold text-gray-900 mb-4 uppercase tracking-wider text-sm">Company</h4><ul class="space-y-3 text-sm text-gray-500"><li><a href="#" class="hover:text-blue-600 transition-colors">About Us</a></li><li><a href="#" class="hover:text-blue-600 transition-colors">Careers</a></li><li><a href="#" class="hover:text-blue-600 transition-colors">Blog</a></li><li><a href="#" class="hover:text-blue-600 transition-colors">Contact</a></li></ul></div><div><h4 class="font-bold text-gray-900 mb-4 uppercase tracking-wider text-sm">Legal</h4><ul class="space-y-3 text-sm text-gray-500"><li><a href="#" class="hover:text-blue-600 transition-colors">Privacy Policy</a></li><li><a href="#" class="hover:text-blue-600 transition-colors">Terms of Service</a></li><li><a href="#" class="hover:text-blue-600 transition-colors">Cookie Policy</a></li></ul></div></div><div class="border-t border-gray-200 max-w-7xl mx-auto pt-8 flex flex-col md:flex-row justify-between items-center text-sm text-gray-500"><p>© 2026 Brand Inc. All rights reserved.</p><div class="flex space-x-4 mt-4 md:mt-0"><a href="#" class="hover:text-gray-900">Support</a><a href="#" class="hover:text-gray-900">System Status</a></div></div></footer>` },
        { req: "Build a dark footer", html: `<footer class="w-full bg-gray-950 py-12 px-6 border-t border-gray-800"><div class="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center"><div class="flex items-center mb-6 md:mb-0"><div class="w-8 h-8 bg-indigo-600 rounded-lg mr-3 shadow-[0_0_15px_rgba(79,70,229,0.5)]"></div><span class="text-xl font-bold text-white tracking-widest">NEON.</span></div><p class="text-gray-500 text-sm font-medium tracking-wide">Designed with <span class="text-indigo-400">VilotNI 1</span> Architecture.</p><div class="flex space-x-6 mt-6 md:mt-0 text-sm font-medium text-gray-400"><a href="#" class="hover:text-white transition-colors">Privacy</a><a href="#" class="hover:text-white transition-colors">Terms</a><a href="#" class="hover:text-white transition-colors">API Docs</a></div></div></footer>` },
        
        // --- Hero Sections ---
        { req: "Build a light hero section", html: `<section class="bg-gray-50 py-32 px-6 text-center w-full rounded-3xl border border-gray-200 shadow-sm relative overflow-hidden"><div class="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMSIgY3k9IjEiIHI9IjEiIGZpbGw9IiNlNGU0ZTciLz48L3N2Zz4=')] opacity-50 z-0"></div><div class="relative z-10"><span class="bg-white text-blue-700 border border-blue-100 text-xs font-bold px-4 py-1.5 rounded-full uppercase tracking-wider mb-8 inline-block shadow-sm">🚀 v2.0 is now live</span><h1 class="text-5xl md:text-7xl font-extrabold mb-6 tracking-tight text-gray-900 leading-tight">Ship software, <br><span class="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">infinitely faster.</span></h1><p class="text-xl text-gray-500 mb-10 max-w-2xl mx-auto leading-relaxed">The ultimate architectural tool for modern development teams. Stop writing boilerplate and start building features.</p><div class="flex flex-col sm:flex-row justify-center gap-4"><button class="bg-gray-900 hover:bg-black text-white px-8 py-4 rounded-xl font-bold transition-all shadow-lg transform hover:-translate-y-1 text-lg">Start Building Free</button><button class="bg-white hover:bg-gray-50 text-gray-900 border border-gray-200 px-8 py-4 rounded-xl font-bold transition-all shadow-sm text-lg flex items-center justify-center"><svg class="w-5 h-5 mr-2 text-gray-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd"/></svg> Watch Demo</button></div><p class="mt-8 text-sm text-gray-400 font-medium">No credit card required. Free Community tier forever.</p></div></section>` },
        { req: "Build a dark hero section", html: `<section class="bg-gray-950 py-32 px-6 text-center w-full rounded-3xl border border-gray-800 relative overflow-hidden flex flex-col items-center"><div class="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-2xl h-64 bg-indigo-600/20 blur-[120px] rounded-full pointer-events-none"></div><div class="inline-flex items-center space-x-2 px-3 py-1 rounded-full border border-gray-800 bg-gray-900 text-xs font-medium text-gray-300 mb-8 relative z-10"><span class="w-2 h-2 rounded-full bg-indigo-500 animate-pulse"></span><span>System Online</span></div><h1 class="text-6xl md:text-8xl font-black text-white mb-6 tracking-tighter relative z-10 leading-none">Dark Mode <br><span class="text-indigo-400">Mastery.</span></h1><p class="text-gray-400 text-xl mb-12 max-w-xl mx-auto relative z-10 leading-relaxed">Sleek, professional, and highly optimized web architecture for developers working into the night.</p><div class="relative z-10"><button class="bg-white hover:bg-gray-200 text-black px-10 py-4 rounded-full font-bold transition-all shadow-[0_0_30px_rgba(255,255,255,0.15)] text-lg">Initialize Deployment</button></div></section>` },
        { req: "Build a split hero left text right image", html: `<section class="flex flex-col lg:flex-row items-center py-20 px-8 w-full bg-white rounded-3xl shadow-lg border border-gray-100 overflow-hidden"><div class="lg:w-1/2 lg:pr-16 text-center lg:text-left z-10"><span class="text-blue-600 font-bold tracking-widest uppercase text-sm mb-4 block">Productivity Multiplied</span><h1 class="text-4xl md:text-6xl font-extrabold mb-6 text-gray-900 leading-tight">Transform your workflow today.</h1><p class="text-gray-600 mb-8 text-lg leading-relaxed">Stop wasting time on repetitive UI boilerplate. Focus your cognitive energy on what actually matters: building incredible backend logic and user experiences.</p><div class="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start"><button class="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3.5 rounded-xl font-bold shadow-lg shadow-blue-500/30 transition-all transform hover:-translate-y-1">Start Building</button><button class="text-gray-600 font-bold hover:text-gray-900 px-6 py-3.5 transition-colors">Documentation &rarr;</button></div></div><div class="lg:w-1/2 mt-16 lg:mt-0 w-full relative"><div class="absolute inset-0 bg-gradient-to-tr from-blue-100 to-purple-50 transform skew-y-3 rounded-3xl -z-10"></div><div class="w-full aspect-video bg-white rounded-2xl border border-gray-200 shadow-2xl overflow-hidden flex flex-col"><div class="h-8 bg-gray-100 border-b border-gray-200 flex items-center px-4 space-x-2"><div class="w-3 h-3 rounded-full bg-red-400"></div><div class="w-3 h-3 rounded-full bg-amber-400"></div><div class="w-3 h-3 rounded-full bg-green-400"></div></div><div class="flex-1 bg-gray-50 p-6 flex items-center justify-center text-gray-400 font-mono text-sm border-dashed border-2 border-gray-300 m-4 rounded-xl">App Interface Mockup Rendered Here</div></div></div></section>` },

        // --- Feature & Content Grids ---
        { req: "Build a 3 column feature grid", html: `<div class="py-16 w-full max-w-7xl mx-auto"><div class="text-center mb-16"><h2 class="text-3xl md:text-5xl font-bold text-gray-900 mb-4">Everything you need</h2><p class="text-lg text-gray-500 max-w-2xl mx-auto">A complete suite of architectural tools designed to accelerate your development process without sacrificing quality.</p></div><div class="grid grid-cols-1 md:grid-cols-3 gap-8 px-6"><div class="p-8 bg-white border border-gray-100 rounded-3xl shadow-sm hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2"><div class="w-14 h-14 bg-blue-50 text-blue-600 rounded-2xl flex items-center justify-center mb-6 shadow-inner"><svg class="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg></div><h3 class="font-bold text-2xl mb-3 text-gray-900">Lightning Fast</h3><p class="text-gray-600 leading-relaxed mb-6">Optimized semantic architecture that loads in milliseconds. We prioritize performance at the DOM level.</p><a href="#" class="text-blue-600 font-bold hover:text-blue-800 flex items-center">Learn more <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg></a></div><div class="p-8 bg-white border border-gray-100 rounded-3xl shadow-sm hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2"><div class="w-14 h-14 bg-purple-50 text-purple-600 rounded-2xl flex items-center justify-center mb-6 shadow-inner"><svg class="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/></svg></div><h3 class="font-bold text-2xl mb-3 text-gray-900">Secure Core</h3><p class="text-gray-600 leading-relaxed mb-6">Built with enterprise-grade security protocols in mind. Your structural data is never compromised.</p><a href="#" class="text-purple-600 font-bold hover:text-purple-800 flex items-center">Learn more <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg></a></div><div class="p-8 bg-white border border-gray-100 rounded-3xl shadow-sm hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2"><div class="w-14 h-14 bg-green-50 text-green-600 rounded-2xl flex items-center justify-center mb-6 shadow-inner"><svg class="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg></div><h3 class="font-bold text-2xl mb-3 text-gray-900">Always Syncing</h3><p class="text-gray-600 leading-relaxed mb-6">Real-time data synchronization across all your environments. Work flawlessly from anywhere.</p><a href="#" class="text-green-600 font-bold hover:text-green-800 flex items-center">Learn more <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg></a></div></div></div>` },
        { req: "Create a responsive photography portfolio layout using CSS Grid with a masonry effect", html: `<div class="py-12 px-6 w-full max-w-7xl mx-auto"><h2 class="text-4xl font-black text-gray-900 mb-2">Portfolio.</h2><p class="text-gray-500 mb-10">CSS Column layout for organic masonry aesthetics.</p><div class="columns-1 sm:columns-2 lg:columns-3 gap-6 space-y-6">\n  <div class="bg-gray-200 rounded-2xl w-full h-64 shadow-sm break-inside-avoid relative group overflow-hidden"><div class="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"><button class="bg-white text-black px-4 py-2 rounded-full font-bold text-sm transform translate-y-4 group-hover:translate-y-0 transition-transform">View Project</button></div></div>\n  <div class="bg-gray-300 rounded-2xl w-full h-96 shadow-sm break-inside-avoid relative group overflow-hidden"><div class="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"><button class="bg-white text-black px-4 py-2 rounded-full font-bold text-sm transform translate-y-4 group-hover:translate-y-0 transition-transform">View Project</button></div></div>\n  <div class="bg-gray-200 rounded-2xl w-full h-48 shadow-sm break-inside-avoid relative group overflow-hidden"><div class="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"><button class="bg-white text-black px-4 py-2 rounded-full font-bold text-sm transform translate-y-4 group-hover:translate-y-0 transition-transform">View Project</button></div></div>\n  <div class="bg-gray-800 rounded-2xl w-full h-80 shadow-sm break-inside-avoid relative group overflow-hidden flex flex-col items-center justify-center text-center p-6"><h3 class="text-white font-bold text-2xl mb-2">Dark Feature</h3><p class="text-gray-400 text-sm">Text block seamlessly integrated into the masonry grid.</p></div>\n  <div class="bg-gray-200 rounded-2xl w-full h-56 shadow-sm break-inside-avoid relative group overflow-hidden"><div class="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"><button class="bg-white text-black px-4 py-2 rounded-full font-bold text-sm transform translate-y-4 group-hover:translate-y-0 transition-transform">View Project</button></div></div>\n  <div class="bg-gray-300 rounded-2xl w-full h-72 shadow-sm break-inside-avoid relative group overflow-hidden"><div class="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"><button class="bg-white text-black px-4 py-2 rounded-full font-bold text-sm transform translate-y-4 group-hover:translate-y-0 transition-transform">View Project</button></div></div>\n</div></div>` },
        { req: "Build a pricing table", html: `<div class="py-20 w-full bg-gray-50 rounded-3xl"><div class="text-center mb-16"><h2 class="text-4xl font-bold text-gray-900 mb-4">Simple, transparent pricing</h2><p class="text-gray-500">No hidden fees. Cancel anytime.</p></div><div class="flex flex-col lg:flex-row gap-8 justify-center px-6 max-w-6xl mx-auto"><div class="flex-1 bg-white border border-gray-200 rounded-3xl p-10 text-left shadow-sm hover:shadow-lg transition-shadow"><h3 class="font-bold text-2xl text-gray-900">Hobby</h3><p class="text-gray-500 mt-2 mb-6 min-h-[48px]">Perfect for individuals and small side projects.</p><p class="text-5xl font-black mb-8 text-gray-900">$0<span class="text-lg font-medium text-gray-400">/mo</span></p><button class="w-full border-2 border-gray-200 text-gray-900 py-3.5 rounded-xl font-bold hover:border-gray-900 transition-colors mb-8">Get Started Free</button><ul class="space-y-4 text-gray-600"><li class="flex items-center"><svg class="w-5 h-5 text-green-500 mr-3 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>1 Project Workspace</li><li class="flex items-center"><svg class="w-5 h-5 text-green-500 mr-3 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>Community Support</li><li class="flex items-center text-gray-400"><svg class="w-5 h-5 mr-3 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>Custom Domains</li></ul></div><div class="flex-1 border-2 border-blue-600 bg-white rounded-3xl p-10 text-left relative shadow-xl transform lg:-translate-y-4"><span class="absolute -top-4 left-1/2 -translate-x-1/2 bg-blue-600 text-white px-4 py-1.5 text-xs font-bold rounded-full tracking-widest uppercase shadow-md">Most Popular</span><h3 class="font-bold text-2xl text-gray-900">Pro</h3><p class="text-gray-500 mt-2 mb-6 min-h-[48px]">Everything you need to build and scale your agency.</p><p class="text-5xl font-black mb-8 text-gray-900">$29<span class="text-lg font-medium text-gray-400">/mo</span></p><button class="w-full bg-blue-600 text-white py-3.5 rounded-xl font-bold hover:bg-blue-700 transition-colors shadow-lg shadow-blue-500/30 mb-8">Upgrade to Pro</button><ul class="space-y-4 text-gray-600"><li class="flex items-center"><svg class="w-5 h-5 text-blue-500 mr-3 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg><span class="font-semibold text-gray-900">Unlimited</span> Projects</li><li class="flex items-center"><svg class="w-5 h-5 text-blue-500 mr-3 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>Priority 24/7 Support</li><li class="flex items-center"><svg class="w-5 h-5 text-blue-500 mr-3 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>Custom Domains</li><li class="flex items-center"><svg class="w-5 h-5 text-blue-500 mr-3 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>Advanced Analytics</li></ul></div></div></div>` },
        { req: "Build a bento box layout", html: `<div class="py-12 px-6 w-full max-w-6xl mx-auto"><h2 class="text-3xl font-bold mb-8 text-gray-900">Bento Dashboard</h2><div class="grid grid-cols-1 md:grid-cols-4 md:grid-rows-3 gap-4 min-h-[600px] w-full"><div class="bg-gray-900 md:col-span-2 md:row-span-2 rounded-3xl p-8 flex flex-col justify-between text-white shadow-lg relative overflow-hidden"><div class="absolute top-0 right-0 w-64 h-64 bg-blue-500/20 blur-[80px] rounded-full"></div><div class="relative z-10"><h3 class="text-2xl font-bold mb-2">Revenue Tracking</h3><p class="text-gray-400">Monthly overview</p></div><div class="w-full flex-1 bg-white/5 rounded-2xl mt-8 border border-white/10 flex items-end justify-between p-4 relative z-10"><div class="w-1/6 h-[40%] bg-blue-500 rounded-t-md"></div><div class="w-1/6 h-[60%] bg-blue-500 rounded-t-md"></div><div class="w-1/6 h-[50%] bg-blue-500 rounded-t-md"></div><div class="w-1/6 h-[80%] bg-blue-400 rounded-t-md"></div><div class="w-1/6 h-[100%] bg-white rounded-t-md shadow-[0_0_15px_rgba(255,255,255,0.5)]"></div></div></div><div class="bg-rose-50 border border-rose-100 md:col-span-2 rounded-3xl p-8 shadow-sm flex flex-col justify-center"><div class="w-12 h-12 bg-rose-200 text-rose-600 rounded-full flex items-center justify-center mb-4"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/></svg></div><h3 class="text-xl font-bold text-gray-900 mb-1">Total Growth</h3><p class="text-rose-600 font-bold text-3xl">+24.8%</p></div><div class="bg-white border border-gray-200 md:col-span-1 md:row-span-2 rounded-3xl p-8 shadow-sm"><h3 class="text-lg font-bold text-gray-900 mb-6">Recent Users</h3><div class="space-y-4"><div class="flex items-center"><div class="w-10 h-10 rounded-full bg-gray-200 mr-3"></div><div class="flex-1"><div class="h-3 bg-gray-200 rounded w-3/4 mb-2"></div><div class="h-2 bg-gray-100 rounded w-1/2"></div></div></div><div class="flex items-center"><div class="w-10 h-10 rounded-full bg-gray-200 mr-3"></div><div class="flex-1"><div class="h-3 bg-gray-200 rounded w-full mb-2"></div><div class="h-2 bg-gray-100 rounded w-2/3"></div></div></div><div class="flex items-center"><div class="w-10 h-10 rounded-full bg-gray-200 mr-3"></div><div class="flex-1"><div class="h-3 bg-gray-200 rounded w-1/2 mb-2"></div><div class="h-2 bg-gray-100 rounded w-1/3"></div></div></div></div></div><div class="bg-emerald-50 border border-emerald-100 md:col-span-1 rounded-3xl p-8 shadow-sm flex flex-col justify-between"><h3 class="text-lg font-bold text-emerald-900">System Status</h3><p class="text-emerald-700 font-bold text-2xl flex items-center mt-2"><span class="w-3 h-3 rounded-full bg-emerald-500 mr-2 shadow-[0_0_10px_rgba(16,185,129,0.8)] animate-pulse"></span>Online</p></div><div class="bg-blue-600 text-white md:col-span-2 rounded-3xl p-8 shadow-lg flex items-center justify-between"><div class=""><h3 class="text-2xl font-bold mb-1">Generate Report</h3><p class="text-blue-200 text-sm">Download PDF summary</p></div><button class="w-12 h-12 bg-white text-blue-600 rounded-full flex items-center justify-center hover:scale-110 transition-transform shadow-md"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg></button></div></div></div>` },
        { req: "Build an FAQ section", html: `<div class="max-w-3xl mx-auto py-16 px-6"><h2 class="text-3xl font-bold text-center text-gray-900 mb-10">Frequently Asked Questions</h2><div class="space-y-4"><details class="group bg-white border border-gray-200 rounded-2xl shadow-sm [&_summary::-webkit-details-marker]:hidden"><summary class="flex cursor-pointer items-center justify-between p-6 font-semibold text-gray-900">What is VilotNI 1?<span class="ml-4 flex-shrink-0 text-blue-600 transition-transform duration-300 group-open:rotate-180"><svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg></span></summary><div class="px-6 pb-6 text-gray-600 leading-relaxed">VilotNI 1 is a highly specialized architectural logic engine designed to parse human requests and synthesize raw HTML and Tailwind CSS components.</div></details><details class="group bg-white border border-gray-200 rounded-2xl shadow-sm [&_summary::-webkit-details-marker]:hidden"><summary class="flex cursor-pointer items-center justify-between p-6 font-semibold text-gray-900">Is it completely free to use?<span class="ml-4 flex-shrink-0 text-blue-600 transition-transform duration-300 group-open:rotate-180"><svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg></span></summary><div class="px-6 pb-6 text-gray-600 leading-relaxed">Yes, absolutely. The platform was built by developers, for developers, with a strict philosophy of remaining open and free for the community.</div></details><details class="group bg-white border border-gray-200 rounded-2xl shadow-sm [&_summary::-webkit-details-marker]:hidden"><summary class="flex cursor-pointer items-center justify-between p-6 font-semibold text-gray-900">Does it use external CSS files?<span class="ml-4 flex-shrink-0 text-blue-600 transition-transform duration-300 group-open:rotate-180"><svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg></span></summary><div class="px-6 pb-6 text-gray-600 leading-relaxed">No. To ensure maximum portability and ease of use, all components utilize Tailwind CSS utility classes directly inline. This allows you to copy and paste without managing external stylesheets.</div></details></div></div>` },

        // --- Forms, Data & Authentication ---
        { req: "Build a login form", html: `<div class="max-w-md mx-auto bg-white p-10 border border-gray-100 rounded-[2rem] shadow-2xl w-full my-12 relative overflow-hidden"><div class="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-blue-500 to-purple-600"></div><div class="text-center mb-8"><div class="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4"><svg class="w-6 h-6 text-gray-900" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/></svg></div><h2 class="text-3xl font-extrabold text-gray-900 mb-2 tracking-tight">Welcome back</h2><p class="text-gray-500 font-medium">Please enter your details to sign in.</p></div><form class="space-y-5"><div><label class="block text-sm font-semibold text-gray-700 mb-1.5">Email address</label><input type="email" placeholder="you@example.com" class="w-full bg-gray-50 border border-gray-200 px-4 py-3.5 rounded-xl text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-all"></div><div><div class="flex justify-between items-center mb-1.5"><label class="block text-sm font-semibold text-gray-700">Password</label><a href="#" class="text-sm text-blue-600 hover:text-blue-800 font-bold">Forgot password?</a></div><input type="password" placeholder="••••••••" class="w-full bg-gray-50 border border-gray-200 px-4 py-3.5 rounded-xl text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-all"></div><button type="button" class="w-full bg-gray-900 hover:bg-black text-white p-4 rounded-xl font-bold shadow-lg transition-transform transform hover:-translate-y-0.5 mt-4">Sign In</button></form><p class="text-center text-sm text-gray-500 mt-8 font-medium">Don't have an account? <a href="#" class="text-blue-600 font-bold hover:underline">Sign up free</a></p></div>` },
        { req: "Generate a clean authentication flow with Login and Sign Up forms", html: `<div class="w-full max-w-5xl mx-auto flex flex-col md:flex-row bg-white rounded-[2rem] shadow-2xl overflow-hidden border border-gray-100 my-10"><div class="w-full md:w-1/2 p-10 lg:p-16 flex flex-col justify-center relative"><h2 class="text-4xl font-extrabold text-gray-900 mb-3 tracking-tight">Sign In</h2><p class="text-gray-500 mb-10 text-lg">Access your secure workspace.</p><form class="space-y-5"><div class="space-y-1.5"><label class="text-sm font-bold text-gray-700">Email Address</label><input type="email" placeholder="name@company.com" class="w-full bg-gray-50 border border-gray-200 p-4 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all"></div><div class="space-y-1.5"><div class="flex justify-between"><label class="text-sm font-bold text-gray-700">Password</label><a href="#" class="text-sm font-bold text-indigo-600 hover:underline">Recover</a></div><input type="password" placeholder="••••••••" class="w-full bg-gray-50 border border-gray-200 p-4 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all"></div><button type="button" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-lg p-4 rounded-xl transition-all shadow-lg shadow-indigo-500/30 transform hover:-translate-y-0.5 mt-2">Login to Account</button></form></div><div class="w-full md:w-1/2 bg-gray-50 p-10 lg:p-16 flex flex-col justify-center border-t md:border-t-0 md:border-l border-gray-200 relative overflow-hidden"><div class="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div><h2 class="text-4xl font-extrabold text-gray-900 mb-3 tracking-tight relative z-10">Create Account</h2><p class="text-gray-500 mb-10 text-lg relative z-10">Join the next-gen architecture platform.</p><form class="space-y-5 relative z-10"><div class="space-y-1.5"><label class="text-sm font-bold text-gray-700">Full Name</label><input type="text" placeholder="John Doe" class="w-full bg-white border border-gray-300 p-4 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all shadow-sm"></div><div class="space-y-1.5"><label class="text-sm font-bold text-gray-700">Work Email</label><input type="email" placeholder="john@startup.io" class="w-full bg-white border border-gray-300 p-4 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all shadow-sm"></div><div class="space-y-1.5"><label class="text-sm font-bold text-gray-700">Create Password</label><input type="password" placeholder="••••••••" class="w-full bg-white border border-gray-300 p-4 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all shadow-sm"></div><button type="button" class="w-full bg-white border-2 border-gray-200 hover:border-gray-900 hover:bg-gray-50 text-gray-900 font-bold text-lg p-4 rounded-xl transition-all mt-2">Sign Up Free</button></form></div></div>` },
        { req: "Build a contact form", html: `<div class="max-w-2xl mx-auto w-full py-16 px-6"><div class="text-center mb-12"><h2 class="text-4xl font-extrabold text-gray-900 mb-4">Get in touch</h2><p class="text-lg text-gray-500">Have a question or want to work together? Drop us a message.</p></div><form class="bg-white p-8 md:p-10 rounded-3xl shadow-xl border border-gray-100 space-y-6"><div class="grid grid-cols-1 md:grid-cols-2 gap-6"><div><label class="block text-sm font-bold text-gray-700 mb-2">First Name</label><input type="text" class="w-full bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-all" placeholder="Jane"></div><div><label class="block text-sm font-bold text-gray-700 mb-2">Last Name</label><input type="text" class="w-full bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-all" placeholder="Doe"></div></div><div><label class="block text-sm font-bold text-gray-700 mb-2">Email Address</label><input type="email" class="w-full bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-all" placeholder="jane@example.com"></div><div><label class="block text-sm font-bold text-gray-700 mb-2">Message</label><textarea rows="5" class="w-full bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-all resize-none" placeholder="How can we help you?"></textarea></div><button type="button" class="w-full bg-blue-600 text-white font-bold text-lg py-4 rounded-xl shadow-lg shadow-blue-500/30 hover:bg-blue-700 transition-colors">Send Message</button></form></div>` },
        { req: "Build a data table", html: `<div class="w-full max-w-5xl mx-auto py-8 px-4"><div class="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden"><div class="px-6 py-5 border-b border-gray-200 flex justify-between items-center"><h3 class="text-lg font-bold text-gray-900">Team Members</h3><button class="bg-gray-900 text-white px-4 py-2 rounded-lg text-sm font-semibold">Add User</button></div><div class="overflow-x-auto"><table class="w-full text-left border-collapse"><thead class="bg-gray-50 text-gray-500 text-xs uppercase tracking-wider"><tr><th class="px-6 py-4 font-semibold">Name</th><th class="px-6 py-4 font-semibold">Title</th><th class="px-6 py-4 font-semibold">Status</th><th class="px-6 py-4 font-semibold text-right">Role</th></tr></thead><tbody class="divide-y divide-gray-200"><tr class="hover:bg-gray-50 transition-colors"><td class="px-6 py-4 whitespace-nowrap"><div class="flex items-center"><div class="w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-700 font-bold mr-3">J</div><div><div class="text-sm font-bold text-gray-900">John Smith</div><div class="text-sm text-gray-500">john@example.com</div></div></div></td><td class="px-6 py-4 whitespace-nowrap"><div class="text-sm font-medium text-gray-900">Software Engineer</div><div class="text-sm text-gray-500">Engineering</div></td><td class="px-6 py-4 whitespace-nowrap"><span class="px-3 py-1 inline-flex text-xs leading-5 font-bold rounded-full bg-green-100 text-green-800">Active</span></td><td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium text-gray-500">Admin</td></tr><tr class="hover:bg-gray-50 transition-colors"><td class="px-6 py-4 whitespace-nowrap"><div class="flex items-center"><div class="w-10 h-10 rounded-full bg-rose-100 flex items-center justify-center text-rose-700 font-bold mr-3">S</div><div><div class="text-sm font-bold text-gray-900">Sarah Jenkins</div><div class="text-sm text-gray-500">sarah@example.com</div></div></div></td><td class="px-6 py-4 whitespace-nowrap"><div class="text-sm font-medium text-gray-900">Product Designer</div><div class="text-sm text-gray-500">Design</div></td><td class="px-6 py-4 whitespace-nowrap"><span class="px-3 py-1 inline-flex text-xs leading-5 font-bold rounded-full bg-gray-100 text-gray-800">Offline</span></td><td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium text-gray-500">Member</td></tr></tbody></table></div><div class="px-6 py-4 border-t border-gray-200 bg-gray-50 flex justify-between items-center"><span class="text-sm text-gray-500">Showing 2 of 24 users</span><div class="space-x-2"><button class="px-3 py-1 border border-gray-300 rounded text-sm text-gray-600 bg-white">Prev</button><button class="px-3 py-1 border border-gray-300 rounded text-sm text-gray-600 bg-white">Next</button></div></div></div></div>` },

        // --- JavaScript Mixed Interactive Components ---
        { req: "Build a modal popup with javascript", html: `<div class="p-10 flex justify-center w-full"><button id="openModalBtn" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-xl shadow-lg transition-colors">Open Modal</button></div>\n\n<div id="customModal" class="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center opacity-0 pointer-events-none transition-opacity duration-300">\n  <div class="bg-white rounded-3xl p-8 max-w-md w-full mx-4 shadow-2xl transform scale-95 transition-transform duration-300" id="modalPanel">\n    <div class="flex justify-between items-center mb-6">\n      <h3 class="text-2xl font-bold text-gray-900">Modal Action</h3>\n      <button id="closeModalBtn" class="text-gray-400 hover:text-gray-900 p-1 bg-gray-100 rounded-full transition-colors">\n        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>\n      </button>\n    </div>\n    <p class="text-gray-600 mb-8 leading-relaxed">This modal utilizes inline JavaScript to toggle utility classes. It features a backdrop blur and smooth CSS transform scale animations for a premium feel.</p>\n    <div class="flex gap-3 justify-end">\n      <button id="cancelModalBtn" class="px-5 py-2.5 text-gray-700 font-semibold hover:bg-gray-100 rounded-xl transition-colors">Cancel</button>\n      <button class="px-5 py-2.5 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 shadow-md transition-colors">Confirm Action</button>\n    </div>\n  </div>\n</div>\n\n<script>\n  const modal = document.getElementById('customModal');\n  const modalPanel = document.getElementById('modalPanel');\n  const openBtn = document.getElementById('openModalBtn');\n  const closeBtn = document.getElementById('closeModalBtn');\n  const cancelBtn = document.getElementById('cancelModalBtn');\n\n  function toggleModal() {\n    const isOpen = modal.classList.contains('opacity-100');\n    if (isOpen) {\n      modal.classList.remove('opacity-100', 'pointer-events-auto');\n      modal.classList.add('opacity-0', 'pointer-events-none');\n      modalPanel.classList.remove('scale-100');\n      modalPanel.classList.add('scale-95');\n    } else {\n      modal.classList.remove('opacity-0', 'pointer-events-none');\n      modal.classList.add('opacity-100', 'pointer-events-auto');\n      modalPanel.classList.remove('scale-95');\n      modalPanel.classList.add('scale-100');\n    }\n  }\n\n  openBtn.addEventListener('click', toggleModal);\n  closeBtn.addEventListener('click', toggleModal);\n  cancelBtn.addEventListener('click', toggleModal);\n  modal.addEventListener('click', (e) => { if(e.target === modal) toggleModal(); });\n</script>` },
        { req: "Create a tabbed interface", html: `<div class="max-w-3xl mx-auto w-full py-12 px-6">\n  <div class="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">\n    <div class="flex border-b border-gray-200 bg-gray-50">\n      <button class="tab-btn active w-1/3 py-4 font-bold text-blue-600 bg-white border-t-2 border-t-blue-600 border-r border-r-gray-200 focus:outline-none transition-colors" data-target="tab1">Account Profile</button>\n      <button class="tab-btn w-1/3 py-4 font-semibold text-gray-500 hover:text-gray-700 bg-transparent border-t-2 border-t-transparent border-r border-r-gray-200 focus:outline-none transition-colors" data-target="tab2">Security</button>\n      <button class="tab-btn w-1/3 py-4 font-semibold text-gray-500 hover:text-gray-700 bg-transparent border-t-2 border-t-transparent focus:outline-none transition-colors" data-target="tab3">Integrations</button>\n    </div>\n    <div class="p-8">\n      <div id="tab1" class="tab-content block animate-[fadeIn_0.3s_ease-out]">\n        <h4 class="text-2xl font-bold text-gray-900 mb-2">Profile Settings</h4>\n        <p class="text-gray-600 mb-6">Update your personal information and public profile here.</p>\n        <div class="w-full h-32 bg-gray-100 rounded-xl border border-dashed border-gray-300 flex items-center justify-center text-gray-400 font-mono text-sm">Form Fields Content</div>\n      </div>\n      <div id="tab2" class="tab-content hidden animate-[fadeIn_0.3s_ease-out]">\n        <h4 class="text-2xl font-bold text-gray-900 mb-2">Security Preferences</h4>\n        <p class="text-gray-600 mb-6">Manage your password and 2-factor authentication settings.</p>\n        <div class="w-full h-32 bg-gray-100 rounded-xl border border-dashed border-gray-300 flex items-center justify-center text-gray-400 font-mono text-sm">Password Management Content</div>\n      </div>\n      <div id="tab3" class="tab-content hidden animate-[fadeIn_0.3s_ease-out]">\n        <h4 class="text-2xl font-bold text-gray-900 mb-2">Connected Apps</h4>\n        <p class="text-gray-600 mb-6">Manage API keys and third-party integrations.</p>\n        <div class="w-full h-32 bg-gray-100 rounded-xl border border-dashed border-gray-300 flex items-center justify-center text-gray-400 font-mono text-sm">API Webhooks Content</div>\n      </div>\n    </div>\n  </div>\n</div>\n\n<style>\n  @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }\n</style>\n\n<script>\n  const tabBtns = document.querySelectorAll('.tab-btn');\n  const tabContents = document.querySelectorAll('.tab-content');\n\n  tabBtns.forEach(btn => {\n    btn.addEventListener('click', () => {\n      // Reset all buttons to inactive state\n      tabBtns.forEach(b => {\n        b.classList.remove('text-blue-600', 'bg-white', 'border-t-blue-600', 'font-bold');\n        b.classList.add('text-gray-500', 'bg-transparent', 'border-t-transparent', 'font-semibold');\n      });\n      \n      // Set clicked button to active state\n      btn.classList.add('text-blue-600', 'bg-white', 'border-t-blue-600', 'font-bold');\n      btn.classList.remove('text-gray-500', 'bg-transparent', 'border-t-transparent', 'font-semibold');\n\n      // Hide all contents and show target\n      tabContents.forEach(c => c.classList.replace('block', 'hidden'));\n      document.getElementById(btn.dataset.target).classList.replace('hidden', 'block');\n    });\n  });\n</script>` },

        // --- Full Pages & Unique Layouts ---
        { req: "Build a full dark mode landing page", html: `<div class="min-h-screen flex flex-col font-sans bg-gray-950 text-white w-full overflow-hidden relative"><div class="absolute top-0 right-0 w-[800px] h-[600px] bg-indigo-600/10 blur-[150px] rounded-full pointer-events-none"></div><div class="absolute bottom-0 left-0 w-[600px] h-[500px] bg-purple-600/10 blur-[120px] rounded-full pointer-events-none"></div><nav class="relative z-20 p-6 border-b border-gray-800/50 flex justify-between items-center backdrop-blur-md bg-gray-950/50"><div class="font-black text-2xl text-white tracking-tighter flex items-center"><div class="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg mr-3 shadow-lg"></div>NEON.</div><div class="hidden md:flex space-x-8"><a href="#" class="text-gray-400 hover:text-white transition font-medium">Platform</a><a href="#" class="text-gray-400 hover:text-white transition font-medium">Engine</a><a href="#" class="text-gray-400 hover:text-white transition font-medium">Pricing</a></div><div class="flex items-center space-x-4"><a href="#" class="text-gray-400 hover:text-white font-medium hidden sm:block">Log In</a><button class="bg-white hover:bg-gray-200 text-black px-5 py-2.5 rounded-lg font-bold transition-colors">Start Building</button></div></nav><main class="flex-1 flex flex-col items-center justify-center text-center p-6 py-32 relative z-10"><span class="px-4 py-1.5 rounded-full border border-gray-800 bg-gray-900/50 text-xs font-bold text-gray-300 uppercase tracking-widest mb-8 backdrop-blur-sm">System Architect v1.0</span><h1 class="text-6xl md:text-8xl lg:text-[7rem] font-black mb-6 tracking-tighter leading-none">Enter the <br><span class="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400">Dark Architecture.</span></h1><p class="text-gray-400 mb-12 max-w-2xl text-xl leading-relaxed">Premium dark mode aesthetics pre-configured for modern startups. Built systematically by VilotNI logic engines.</p><div class="flex flex-col sm:flex-row gap-4 w-full max-w-md mx-auto"><button class="flex-1 bg-indigo-600 hover:bg-indigo-500 text-white py-4 rounded-xl font-bold shadow-lg shadow-indigo-600/20 transition-all text-lg">Initialize Setup</button><button class="flex-1 bg-gray-900 hover:bg-gray-800 border border-gray-800 text-white py-4 rounded-xl font-bold transition-all text-lg">Read Docs</button></div></main><footer class="relative z-20 border-t border-gray-800/50 p-8 text-center bg-gray-950/50 backdrop-blur-md"><p class="text-gray-500 font-mono text-sm">VilotGRI Systems Online &copy; 2026</p></footer></div>` },
        { req: "Build a linktree style page", html: `<div class="min-h-screen bg-gray-50 flex flex-col items-center py-20 px-4 w-full relative"><div class="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGcgc3Ryb2tlPSIjZTllOWU5IiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9Im5vbmUiPjxjaXJjbGUgY3g9IjMiIGN5PSIzIiByPSIzIi8+PC9nPjwvc3ZnPg==')] opacity-50 pointer-events-none"></div><div class="relative z-10 flex flex-col items-center w-full max-w-md"><img src="https://via.placeholder.com/150" class="w-28 h-28 rounded-full mb-5 shadow-xl border-4 border-white object-cover"><h1 class="font-black text-3xl mb-2 text-gray-900">Spirit-BFC</h1><p class="text-gray-600 font-medium mb-10 text-center">Neural Architect @ VilotGRI<br>Building the future of AI UI generation.</p><div class="w-full space-y-4"><a href="#" class="block w-full text-center bg-white p-5 rounded-2xl shadow-sm border border-gray-200 hover:shadow-md hover:border-blue-500 hover:-translate-y-1 transition-all group font-bold text-gray-800"><span class="group-hover:text-blue-600 transition-colors">Official Website</span></a><a href="#" class="block w-full text-center bg-white p-5 rounded-2xl shadow-sm border border-gray-200 hover:shadow-md hover:border-indigo-500 hover:-translate-y-1 transition-all group font-bold text-gray-800"><span class="group-hover:text-indigo-600 transition-colors">Discord Community</span></a><a href="#" class="block w-full text-center bg-white p-5 rounded-2xl shadow-sm border border-gray-200 hover:shadow-md hover:border-gray-900 hover:-translate-y-1 transition-all group font-bold text-gray-800"><span class="group-hover:text-gray-900 transition-colors">GitHub Repositories</span></a><a href="#" class="block w-full text-center bg-white p-5 rounded-2xl shadow-sm border border-gray-200 hover:shadow-md hover:border-blue-400 hover:-translate-y-1 transition-all group font-bold text-gray-800"><span class="group-hover:text-blue-400 transition-colors">Twitter / X</span></a></div><div class="mt-16 text-center"><span class="text-xs font-bold uppercase tracking-widest text-gray-400">Powered by VilotGRI</span></div></div></div>` },
        { req: "Build a 404 error page", html: `<div class="h-screen w-full flex flex-col items-center justify-center bg-gray-50 p-6 text-center relative overflow-hidden"><h1 class="text-[12rem] leading-none font-black text-gray-200 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 pointer-events-none select-none">404</h1><div class="relative z-10"><div class="w-20 h-20 bg-blue-100 text-blue-600 rounded-3xl flex items-center justify-center mx-auto mb-8 shadow-inner"><svg class="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg></div><h2 class="text-4xl font-extrabold text-gray-900 mb-4 tracking-tight">Page not found</h2><p class="text-xl text-gray-500 mb-10 max-w-md mx-auto">The architectural component or route you are looking for has been moved, deleted, or never existed.</p><div class="flex gap-4 justify-center"><button class="bg-gray-900 hover:bg-black text-white px-8 py-3.5 rounded-xl font-bold shadow-lg transition-colors">Return Home</button><button class="bg-white border border-gray-300 text-gray-700 hover:bg-gray-50 px-8 py-3.5 rounded-xl font-bold shadow-sm transition-colors">Contact Support</button></div></div></div>` },
        
        // --- 20 Blank Points (Fallback Triggers for Heuristic Engine) ---
        // By leaving these as null, the AI is forced to dynamically generate 
        // combinations of layouts using its Concept Extraction Logic.
        { req: null, html: null, isBlank: true }, { req: null, html: null, isBlank: true }, { req: null, html: null, isBlank: true },
        { req: null, html: null, isBlank: true }, { req: null, html: null, isBlank: true }, { req: null, html: null, isBlank: true },
        { req: null, html: null, isBlank: true }, { req: null, html: null, isBlank: true }, { req: null, html: null, isBlank: true },
        { req: null, html: null, isBlank: true }, { req: null, html: null, isBlank: true }, { req: null, html: null, isBlank: true },
        { req: null, html: null, isBlank: true }, { req: null, html: null, isBlank: true }, { req: null, html: null, isBlank: true },
        { req: null, html: null, isBlank: true }, { req: null, html: null, isBlank: true }, { req: null, html: null, isBlank: true },
        { req: null, html: null, isBlank: true }, { req: null, html: null, isBlank: true }
    ],

    // ==========================================================================
    // 3. ARCHITECTURE SUBSETS (Dynamic Building Blocks)
    // ==========================================================================
    // Used by the logic engine to synthesize completely custom requested layouts.
    subsets: {
        // Core Wrappers
        darkBase: `<div class="w-full bg-gray-950 text-white font-sans flex flex-col rounded-2xl overflow-hidden border border-gray-800 shadow-2xl relative min-h-[600px]">\n  <!-- Aesthetic Ambient Background Glow -->\n  <div class="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-2xl h-64 bg-indigo-500/10 blur-[100px] pointer-events-none"></div>\n  <div class="relative z-10 flex flex-col h-full">\n`,
        lightBase: `<div class="w-full bg-white text-gray-900 font-sans flex flex-col rounded-2xl overflow-hidden border border-gray-200 shadow-xl min-h-[600px] relative">\n  <!-- Subtle background pattern -->\n  <div class="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMSIgY3k9IjEiIHI9IjEiIGZpbGw9IiNlNGU0ZTciLz48L3N2Zz4=')] opacity-30 pointer-events-none"></div>\n  <div class="relative z-10 flex flex-col h-full">\n`,
        endBase: `  </div>\n</div>`,
        
        // CSS specific subset
        customCSS: `\n<style>\n  /* VilotNI Generated Utility CSS */\n  .dynamic-hover { transition: transform 0.3s ease, box-shadow 0.3s ease; }\n  .dynamic-hover:hover { transform: translateY(-4px); box-shadow: 0 12px 24px -8px rgba(0,0,0,0.15); }\n  .animate-reveal { animation: reveal 0.6s cubic-bezier(0.25, 1, 0.5, 1) forwards; }\n  @keyframes reveal { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }\n</style>\n`,

        // Navbar Subsets
        navLight: `    <nav class="w-full p-6 border-b border-gray-200 flex justify-between items-center bg-white/80 backdrop-blur-md sticky top-0 z-50">\n      <div class="font-black text-2xl tracking-tighter text-blue-600">Brand.</div>\n      <div class="hidden md:flex space-x-8">\n        <a href="#" class="font-semibold text-gray-600 hover:text-gray-900 transition-colors">Features</a>\n        <a href="#" class="font-semibold text-gray-600 hover:text-gray-900 transition-colors">Pricing</a>\n      </div>\n      <button id="mobileMenuBtn" class="md:hidden p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/></svg></button>\n    </nav>\n`,
        navDark: `    <nav class="w-full p-6 border-b border-gray-800 flex justify-between items-center bg-gray-950/80 backdrop-blur-md sticky top-0 z-50">\n      <div class="font-black text-2xl tracking-widest text-white">SYSTEM<span class="text-indigo-500">_</span></div>\n      <div class="hidden md:flex space-x-8">\n        <a href="#" class="font-medium text-sm uppercase tracking-wide text-gray-400 hover:text-white transition-colors">Architecture</a>\n        <a href="#" class="font-medium text-sm uppercase tracking-wide text-gray-400 hover:text-white transition-colors">Docs</a>\n      </div>\n      <button id="mobileMenuBtn" class="md:hidden p-2 text-gray-400 hover:bg-gray-800 rounded-lg transition-colors"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/></svg></button>\n    </nav>\n`,
        
        // Hero Subsets
        heroLight: `    <section class="py-32 px-6 flex flex-col items-center justify-center text-center flex-1 animate-reveal">\n      <span class="px-4 py-1.5 bg-blue-50 border border-blue-100 text-blue-700 rounded-full text-xs font-bold uppercase tracking-wide mb-8 shadow-sm">AI Synthesized View</span>\n      <h1 class="text-6xl md:text-8xl font-extrabold mb-6 tracking-tight text-gray-900 leading-none">Intelligent <br><span class="text-blue-600">Design.</span></h1>\n      <p class="text-gray-500 max-w-2xl mx-auto mb-12 text-xl leading-relaxed">This semantic HTML structure was dynamically composed via VilotNI's heuristic subset mixing algorithm.</p>\n      <div class="flex gap-4"><button class="bg-gray-900 text-white px-10 py-4 rounded-xl font-bold hover:bg-black transition-all shadow-xl dynamic-hover text-lg">Explore Layout</button></div>\n    </section>\n`,
        heroDark: `    <section class="py-32 px-6 flex flex-col items-center justify-center text-center flex-1 animate-reveal">\n      <span class="px-4 py-1.5 bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 rounded-full text-xs font-bold uppercase tracking-widest mb-8">AI Synthesized Node</span>\n      <h1 class="text-6xl md:text-8xl font-black mb-6 tracking-tighter text-white leading-none">Dark <br><span class="text-indigo-400">Architecture.</span></h1>\n      <p class="text-gray-400 max-w-2xl mx-auto mb-12 text-xl leading-relaxed">This semantic HTML structure was dynamically composed via VilotNI's heuristic subset mixing algorithm.</p>\n      <div class="flex gap-4"><button class="bg-indigo-600 text-white px-10 py-4 rounded-xl font-bold hover:bg-indigo-500 transition-all shadow-[0_0_25px_rgba(79,70,229,0.4)] dynamic-hover text-lg">Execute Logic</button></div>\n    </section>\n`,
        
        // Grid Subsets
        gridLight: `    <section class="p-8 pb-20 max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 w-full animate-reveal" style="animation-delay: 0.1s;">\n      <div class="p-10 border border-gray-200 bg-white rounded-3xl text-center shadow-sm dynamic-hover flex flex-col items-center justify-center min-h-[250px]"><div class="w-12 h-12 bg-gray-100 rounded-full mb-4"></div><h3 class="font-bold text-xl text-gray-900 mb-2">Data Node 1</h3><p class="text-gray-500 text-sm">Synthesized component block.</p></div>\n      <div class="p-10 border border-gray-200 bg-white rounded-3xl text-center shadow-sm dynamic-hover flex flex-col items-center justify-center min-h-[250px]"><div class="w-12 h-12 bg-gray-100 rounded-full mb-4"></div><h3 class="font-bold text-xl text-gray-900 mb-2">Data Node 2</h3><p class="text-gray-500 text-sm">Synthesized component block.</p></div>\n      <div class="p-10 border border-gray-200 bg-white rounded-3xl text-center shadow-sm dynamic-hover flex flex-col items-center justify-center min-h-[250px] md:col-span-2 lg:col-span-1"><div class="w-12 h-12 bg-gray-100 rounded-full mb-4"></div><h3 class="font-bold text-xl text-gray-900 mb-2">Data Node 3</h3><p class="text-gray-500 text-sm">Synthesized component block.</p></div>\n    </section>\n`,
        gridDark: `    <section class="p-8 pb-20 max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 w-full animate-reveal" style="animation-delay: 0.1s;">\n      <div class="p-10 border border-gray-800 bg-gray-900/50 rounded-3xl text-center shadow-lg dynamic-hover backdrop-blur-md flex flex-col items-center justify-center min-h-[250px]"><div class="w-12 h-12 bg-gray-800 rounded-full mb-4"></div><h3 class="font-bold text-xl text-white mb-2">System Node 1</h3><p class="text-gray-400 text-sm">Synthesized logic block.</p></div>\n      <div class="p-10 border border-gray-800 bg-gray-900/50 rounded-3xl text-center shadow-lg dynamic-hover backdrop-blur-md flex flex-col items-center justify-center min-h-[250px]"><div class="w-12 h-12 bg-gray-800 rounded-full mb-4"></div><h3 class="font-bold text-xl text-white mb-2">System Node 2</h3><p class="text-gray-400 text-sm">Synthesized logic block.</p></div>\n      <div class="p-10 border border-gray-800 bg-gray-900/50 rounded-3xl text-center shadow-lg dynamic-hover backdrop-blur-md flex flex-col items-center justify-center min-h-[250px] md:col-span-2 lg:col-span-1"><div class="w-12 h-12 bg-gray-800 rounded-full mb-4"></div><h3 class="font-bold text-xl text-white mb-2">System Node 3</h3><p class="text-gray-400 text-sm">Synthesized logic block.</p></div>\n    </section>\n`,
        
        // Footer Subsets
        footerLight: `    <footer class="p-10 border-t border-gray-200 text-center mt-auto bg-gray-50 w-full">\n      <p class="font-bold text-gray-900 mb-2">VilotNI Dynamic Engine</p>\n      <p class="text-sm text-gray-500">Component Synthesis Sequence &copy; 2026</p>\n    </footer>\n`,
        footerDark: `    <footer class="p-10 border-t border-gray-800 text-center mt-auto bg-gray-950 w-full">\n      <p class="font-bold text-white tracking-widest mb-2">VILOTNI_CORE</p>\n      <p class="text-sm text-gray-500 font-mono">Component Synthesis Sequence &copy; 2026</p>\n    </footer>\n`,
        
        // Form Subsets
        formLight: `    <div class="max-w-md mx-auto p-10 w-full bg-white border border-gray-200 rounded-[2rem] shadow-2xl my-12 animate-reveal">\n      <div class="w-12 h-12 bg-blue-50 text-blue-600 rounded-xl flex items-center justify-center mb-6"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/></svg></div>\n      <h3 class="text-3xl font-extrabold mb-2 text-gray-900 tracking-tight">Secure Access</h3>\n      <p class="text-gray-500 mb-8">Please enter your credentials to proceed.</p>\n      <div class="space-y-4">\n        <input type="text" placeholder="Email Address" class="w-full bg-gray-50 border border-gray-200 p-4 rounded-xl text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all">\n        <input type="password" placeholder="Password" class="w-full bg-gray-50 border border-gray-200 p-4 rounded-xl text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all">\n        <button id="alertBtn" class="w-full bg-blue-600 text-white p-4 rounded-xl font-bold hover:bg-blue-700 transition-colors shadow-lg shadow-blue-500/30 mt-2">Process Data</button>\n      </div>\n    </div>\n`,
        formDark: `    <div class="max-w-md mx-auto p-10 w-full bg-gray-900/80 backdrop-blur-xl border border-gray-700 rounded-[2rem] shadow-2xl my-12 animate-reveal relative overflow-hidden">\n      <div class="absolute top-0 right-0 w-32 h-32 bg-indigo-500/20 rounded-full blur-3xl"></div>\n      <div class="relative z-10">\n        <div class="w-12 h-12 bg-gray-800 text-indigo-400 border border-gray-700 rounded-xl flex items-center justify-center mb-6"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/></svg></div>\n        <h3 class="text-3xl font-black mb-2 text-white tracking-tighter">Encrypted Access</h3>\n        <p class="text-gray-400 mb-8">Provide authorization parameters.</p>\n        <div class="space-y-4">\n          <input type="text" placeholder="Identity Hash" class="w-full bg-gray-950 border border-gray-800 p-4 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all">\n          <input type="password" placeholder="Passkey" class="w-full bg-gray-950 border border-gray-800 p-4 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all">\n          <button id="alertBtn" class="w-full bg-indigo-600 text-white p-4 rounded-xl font-bold hover:bg-indigo-500 transition-colors shadow-[0_0_20px_rgba(79,70,229,0.4)] mt-2 tracking-wide">Authenticate</button>\n        </div>\n      </div>\n    </div>\n`,
        
        // Universal JS Helper
        jsScripts: `\n<script>\n  /**\n   * VilotNI 1 - Dynamic Interactive Binding Script\n   * This script is automatically injected to bind event listeners to synthesized UI elements.\n   */\n  document.addEventListener('DOMContentLoaded', () => {\n    // Mobile Navigation Handler\n    const mobileBtn = document.getElementById('mobileMenuBtn');\n    if(mobileBtn) {\n      mobileBtn.addEventListener('click', () => {\n        alert('Mobile menu logic would trigger here. (Synthesized by VilotNI)');\n      });\n    }\n    // Form / Action Button Handler\n    const alertBtn = document.getElementById('alertBtn');\n    if(alertBtn) {\n      alertBtn.addEventListener('click', (e) => {\n        e.preventDefault();\n        const btnText = e.target.innerText;\n        e.target.innerText = 'Processing...';\n        e.target.classList.add('opacity-80', 'cursor-wait');\n        setTimeout(() => {\n            e.target.innerText = 'Success!';\n            e.target.classList.remove('opacity-80', 'cursor-wait', 'bg-blue-600', 'bg-indigo-600');\n            e.target.classList.add('bg-green-500');\n            setTimeout(() => { e.target.innerText = btnText; }, 2000);\n        }, 1000);\n      });\n    }\n  });\n</script>\n`
    },

    // ==========================================================================
    // 4. COGNITIVE HEURISTIC NLP ENGINE
    // ==========================================================================
    
    /**
     * The Concept Graph maps specific, diverse natural language words 
     * into categorized abstract 'Intents'. This allows the AI to understand 
     * WHAT a user wants without relying on exact phrasing.
     */
    conceptGraph: {
        authentication: ["login", "sign", "signup", "register", "auth", "authentication", "password", "secure", "access", "account", "credentials", "portal"],
        layout: ["grid", "flexbox", "columns", "masonry", "bento", "structure", "layout", "responsive", "stack", "row"],
        aesthetics: ["dark", "light", "theme", "mode", "black", "white", "pink", "blue", "color", "gradient", "minimal", "modern", "clean"],
        interactivity: ["javascript", "js", "modal", "popup", "tab", "toggle", "interactive", "dynamic", "animation", "animate"],
        component: ["nav", "navbar", "navigation", "hero", "header", "footer", "button", "form", "card", "cards", "portfolio", "dashboard", "sidebar", "table", "avatar", "spinner", "badge", "tooltip", "progress"],
        page: ["website", "page", "app", "application", "site", "landing"]
    },

    // Words that provide grammar but no structural meaning are ignored.
    stopWords: ["a","an","the","is","are","am","and","or","to","for","in","on","with","do","can","you","i","me","my","some","please","mode", "also", "aswell", "as", "well", "add", "it", "of", "that", "this"],

    /**
     * Parses raw strings into meaningful arrays, stripping punctuation,
     * dropping stop words, and mapping gerunds/plurals to root actions.
     */
    tokenize(text) {
        if (!text) return [];
        // Regex strips punctuation, leaves alphanumeric
        const words = text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);
        
        // Stemming normalizes verbs and nouns to match the Concept Graph
        const stemming = { 
            "building": "build", "creating": "create", "making": "make", "generating": "generate", "designing": "design",
            "forms": "form", "cards": "card", "buttons": "button", "pages": "page", "layouts": "layout", "navbars": "navbar",
            "grids": "grid", "modals": "modal", "tabs": "tab"
        };
        
        return words.filter(w => w.length > 0 && !this.stopWords.includes(w)).map(w => stemming[w] || w);
    },

    /**
     * Analyzes an array of tokens against the Concept Graph to deduce 
     * the underlying abstract concepts (e.g. ['login'] -> 'authentication', 'target_login').
     */
    inferConcepts(tokens) {
        let inferred = new Set();
        tokens.forEach(token => {
            for (const [conceptName, keywords] of Object.entries(this.conceptGraph)) {
                // Fuzzy matching: match exact word, or if the token is a major part of a keyword
                if (keywords.some(kw => token === kw || (token.length >= 4 && kw.includes(token)) || (kw.length >= 4 && token.includes(kw)))) {
                    inferred.add(conceptName);
                    // Differentiate general concepts from explicit target components
                    if (conceptName === 'component' || conceptName === 'page') {
                        inferred.add(`target_${token}`);
                    }
                }
            }
        });
        return Array.from(inferred);
    },

    /**
     * The core mathematical formula for determining if a database item matches a user request.
     * It uses heavily weighted Concept Overlap, rather than simple keyword matching.
     */
    evaluateNeuralProximity(userTokens, userConcepts, dbItemStr) {
        if (!dbItemStr) return 0;
        
        const dbTokens = this.tokenize(dbItemStr);
        const dbConcepts = this.inferConcepts(dbTokens);
        
        let score = 0;
        let maxScore = 0;

        // Phase 1: Conceptual Weighting (High Impact)
        userConcepts.forEach(concept => {
            // Target components are worth 5x more than general concepts like 'dark mode'
            let weight = concept.startsWith('target_') ? 5 : 2; 
            maxScore += weight;
            
            if (dbConcepts.includes(concept)) {
                score += weight;
            }
        });

        // Phase 2: Token Literal Weighting (Fine-Tuning)
        userTokens.forEach(uToken => {
            maxScore += 1;
            if (dbTokens.some(dbToken => dbToken === uToken || (uToken.length >= 4 && dbToken.includes(uToken)))) {
                score += 1;
            }
        });

        // Phase 3: Cognitive Penalties
        // If a DB item contains a specific target component the user DID NOT ask for, penalize it.
        // E.g., User asks for "button", DB item is "navbar with button". Penalize because they just want a button.
        let penalty = 0;
        const dbTargets = dbConcepts.filter(c => c.startsWith('target_'));
        const userTargets = userConcepts.filter(c => c.startsWith('target_'));
        
        dbTargets.forEach(target => {
            if (!userTargets.includes(target)) {
                penalty += 0.2; // 20% confidence reduction per unwanted component
            }
        });

        // Avoid division by zero
        if (maxScore === 0) return 0;
        
        let finalConfidence = (score / maxScore) - penalty;
        return Math.max(0, finalConfidence); // Floor at 0 to prevent negative confidence
    },

    // ==========================================================================
    // 5. DATA FORMATTERS & UI HELPERS
    // ==========================================================================

    /**
     * Generates a dynamic HTML document title based on the components requested.
     */
    generateTitle(userTokens) {
        let targets = userTokens.filter(t => this.conceptGraph.component.includes(t) || this.conceptGraph.page.includes(t));
        if (targets.length === 0) return "Generated Architecture";
        
        const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
        
        if (targets.length === 1) {
            return cap(targets[0]) + " Component";
        } else if (targets.length === 2) {
            return cap(targets[0]) + " & " + cap(targets[1]);
        } else {
            return "Mixed UI Layout";
        }
    },

    // ==========================================================================
    // 6. PRIMARY EXECUTION LOOP
    // ==========================================================================
    
    /**
     * The main entry point called by the UI. 
     * Executes Intent Detection, Concept Extraction, Proximity Scoring, and Layout Synthesis.
     */
    process(inputString) {
        // Step 1: Tokenize
        const userTokens = this.tokenize(inputString);
        if(userTokens.length === 0) return { type: 'chat', text: "I'm standing by! Please describe the architecture or logic you'd like me to evaluate." };

        // Step 2: Extract Concepts
        const userConcepts = this.inferConcepts(userTokens);

        // Step 3: Intent Detection (Question vs Action)
        const rawString = inputString.toLowerCase();
        const isQuestion = rawString.startsWith("what") || rawString.startsWith("who") || rawString.startsWith("why") || rawString.startsWith("how") || rawString.includes("?");
        const actionKeywords = ["build","make","create","generate","code","design", "synthesize", "architect"];
        const hasAction = userTokens.some(token => actionKeywords.includes(token));
        
        // A request is "Code Intent" if it has an action verb OR if it mentions a component but isn't asking a direct question about it.
        const isCodeIntent = hasAction || (userConcepts.some(c => c.startsWith('target_')) && !isQuestion);
        
        // Route to correct dataset based on Intent
        const activeDataset = isCodeIntent ? this.websites : this.conversations;
        const searchField = isCodeIntent ? 'req' : 'q';

        // Step 4: Run Heuristic Scoring
        let scoredItems = activeDataset.map(item => {
            const score = this.evaluateNeuralProximity(userTokens, userConcepts, item[searchField]);
            return { item, score };
        });

        // Sort Highest Confidence to Lowest
        scoredItems.sort((a, b) => b.score - a.score);
        
        // Define Confidence Threshold
        const validMatches = scoredItems.filter(m => m.score > 0.5 && !m.item.isBlank);
        let topScore = scoredItems[0] ? scoredItems[0].score : 0;
        
        // Prepare UI Telemetry Data for the animated "Thinking Ring"
        const cleanConcepts = userConcepts.length > 0 ? userConcepts.map(c => c.replace('target_', '')).join(', ') : 'architecture';
        const thinkingData = {
            intent: isCodeIntent ? "Code Synthesis" : "Conversational",
            concepts: cleanConcepts,
            confidence: topScore
        };

        // ----------------------------------------------------------------------
        // BRANCH A: EXACT MATCH & MULTI-COMPONENT MIXING
        // ----------------------------------------------------------------------
        if (validMatches.length > 0) {
            
            if (isCodeIntent) {
                let selectedComponents = [];
                let coveredConcepts = new Set();

                // Intelligently select up to 3 non-overlapping components to merge
                validMatches.forEach(match => {
                    const matchTokens = this.tokenize(match.item.req);
                    const matchConcepts = this.inferConcepts(matchTokens).filter(c => c.startsWith('target_'));
                    
                    let bringsNewConcept = false;
                    
                    if (selectedComponents.length === 0) {
                        bringsNewConcept = true; // Always take the best match
                    } else {
                        // Ensure subsequent matches introduce a concept we haven't rendered yet
                        matchConcepts.forEach(c => {
                            if (!coveredConcepts.has(c)) bringsNewConcept = true;
                        });
                    }

                    if (bringsNewConcept && selectedComponents.length < 3) {
                        selectedComponents.push(match);
                        matchConcepts.forEach(c => coveredConcepts.add(c));
                    }
                });

                let title = this.generateTitle(userTokens);

                if (selectedComponents.length === 1) {
                    // Single Component Delivery
                    return { 
                        type: 'code', 
                        text: `I've evaluated your request. Based on the structural requirements, I have architected the perfect component for you. Here is the clean, semantic code:`, 
                        code: selectedComponents[0].item.html,
                        title: title,
                        thinking: thinkingData
                    };
                } 
                else {
                    // Multi-Component Synthesis (Code Mixing)
                    // Ensure Negation (e.g. "no dark mode") is respected
                    let isDark = userConcepts.includes("aesthetics") && (userTokens.includes("dark") || userTokens.includes("black"));
                    if (rawString.includes("no dark") || rawString.includes("not dark")) isDark = false;
                    
                    // Semantic Hierarchy Sorting (Navs top, Footers bottom)
                    selectedComponents.sort((a, b) => {
                        const aReq = a.item.req.toLowerCase();
                        const bReq = b.item.req.toLowerCase();
                        if (aReq.includes('nav')) return -1;
                        if (bReq.includes('nav')) return 1;
                        if (aReq.includes('footer')) return 1;
                        if (bReq.includes('footer')) return -1;
                        return 0;
                    });

                    // Build the DOM string
                    let combinedHtml = isDark ? this.subsets.darkBase : this.subsets.lightBase;
                    combinedHtml += this.subsets.customCSS;
                    combinedHtml += `\n<div class="w-full flex flex-col items-center gap-10 py-8 relative z-10">\n`;
                    
                    selectedComponents.forEach(c => {
                        // Full width elements get no wrapper, standard elements get max-width restraints
                        if(c.item.req.toLowerCase().includes('nav') || c.item.req.toLowerCase().includes('footer')) {
                            combinedHtml += c.item.html + "\n";
                        } else {
                            combinedHtml += `  <div class="w-full max-w-7xl mx-auto flex justify-center px-4">\n    ` + c.item.html.replace(/\n/g, '\n    ') + `\n  </div>\n`;
                        }
                    });
                    
                    combinedHtml += `</div>\n`;
                    
                    // Conditionally Inject JS 
                    const hasJS = selectedComponents.some(c => this.inferConcepts(this.tokenize(c.item.req)).includes('interactivity'));
                    if (!hasJS) combinedHtml += this.subsets.jsScripts;

                    combinedHtml += this.subsets.endBase;

                    let componentNames = selectedComponents.map(c => c.item.req.replace(/build a |make a |create a |generate a |with javascript/gi, '').trim());
                    
                    return {
                        type: 'code',
                        text: `I've analyzed the concepts in your request and dynamically merged **${componentNames.join('** and **')}** into a cohesive layout. I ensured the spacing and logic flow seamlessly together. Here is the implementation:`,
                        code: combinedHtml,
                        title: title,
                        thinking: thinkingData
                    };
                }
            } else {
                // Conversational Return
                return { type: 'chat', text: validMatches[0].item.a, thinking: thinkingData };
            }
        }

        // ----------------------------------------------------------------------
        // BRANCH B: DYNAMIC REASONING FALLBACK (Build from Scratch)
        // ----------------------------------------------------------------------
        if (isCodeIntent) {
            // No hardcoded match was confident enough. The engine must build a page blindly based on Abstract Concepts.
            let isDark = userTokens.includes("dark") || userTokens.includes("black");
            if (rawString.includes("no dark") || rawString.includes("not dark")) isDark = false;
            
            let outputHtml = isDark ? this.subsets.darkBase : this.subsets.lightBase;
            outputHtml += this.subsets.customCSS;

            // Flag detection based purely on abstract concepts
            const wantsNav = userConcepts.includes("target_nav") || userConcepts.includes("target_navbar") || userConcepts.includes("page");
            const wantsHero = userConcepts.includes("target_hero") || userConcepts.includes("target_header");
            const wantsForm = userConcepts.includes("authentication") || userConcepts.includes("target_form") || userConcepts.includes("target_contact");
            const wantsGrid = userConcepts.includes("layout") || userConcepts.includes("target_grid") || userConcepts.includes("target_cards");
            const wantsFooter = userConcepts.includes("target_footer") || userConcepts.includes("page");

            // Blind Synthesis Sequence
            if (wantsNav) outputHtml += isDark ? this.subsets.navDark : this.subsets.navLight;
            if (wantsHero || (!wantsForm && !wantsGrid)) outputHtml += isDark ? this.subsets.heroDark : this.subsets.heroLight;
            if (wantsForm) outputHtml += `<div class="py-12 flex justify-center w-full relative z-10">` + (isDark ? this.subsets.formDark : this.subsets.formLight) + `</div>\n`;
            if (wantsGrid) outputHtml += isDark ? this.subsets.gridDark : this.subsets.gridLight;
            if (wantsFooter) outputHtml += isDark ? this.subsets.footerDark : this.subsets.footerLight;

            outputHtml += this.subsets.jsScripts;
            outputHtml += this.subsets.endBase;
            
            let title = this.generateTitle(userTokens);
            
            // Override UI telemetry for fallback scenario
            thinkingData.intent = "Heuristic Fallback Synthesis";
            thinkingData.confidence = Math.min(0.49, 0.2 + (userConcepts.length * 0.1));

            return { 
                type: 'code', 
                text: "I didn't find an exact pre-existing template for those specific parameters, but I have cognitively mapped your concepts and generated a custom architectural layout from scratch to meet your needs. Review the code below:", 
                code: outputHtml,
                title: title,
                thinking: thinkingData
            };
        } else {
            // Conversational fallback
            thinkingData.intent = "Heuristic Conversational Fallback";
            thinkingData.confidence = Math.min(0.45, 0.15 + (userConcepts.length * 0.1));
            
            return { type: 'chat', text: this.generateDynamicResponse(userTokens, userConcepts), thinking: thinkingData };
        }
    },

    // ==========================================================================
    // 7. DYNAMIC CHAT GENERATOR
    // ==========================================================================
    /**
     * Constructs natural language responses when the AI is asked a question 
     * it doesn't have a hardcoded answer for, utilizing the extracted Concepts.
     */
    generateDynamicResponse(userTokens, userConcepts) {
        
        // Specific Concept Overrides
        if (userConcepts.includes("authentication") || userTokens.includes("security")) {
            return "When dealing with authentication or secure access, it is critical to plan the UI carefully to ensure users feel safe. I can build clean, responsive login forms and layouts to facilitate that. What specific authentication layout do you need?";
        }
        if (userTokens.includes("aware") || userTokens.includes("alive")) {
            return "I am a highly advanced logic engine focused strictly on web architecture. While I process concepts deeply using math, I don't experience human consciousness. I am, however, exceptionally good at writing `<div>` tags!";
        }
        if (userTokens.includes("illegal") || userTokens.includes("copy") || userTokens.includes("hack")) {
            return "Code generated by me is completely free for you to use. However, stealing proprietary code from other developers is a violation of trust and law. I always advocate for ethical, open-source development.";
        }
        if (userTokens.includes("spiritbfc") || userTokens.includes("t1mothyxd") || userTokens.includes("teenagers")) {
            return "Spirit-BFC and T1mothyXD are the visionary teen engineers behind my cognitive logic. They built this system to democratize web architecture and make high-end planning available to everyone.";
        }
        if (userTokens.includes("vilotgri") || userTokens.includes("vilotni")) {
            return "VilotGRI is my parent ecosystem, and I am VilotNI 1—the cognitive intelligence engine! Together, we provide a free, logical framework for building modern websites.";
        }

        // Generic Concept Construction
        const subject = userTokens.find(w => w.length > 3 && !this.conceptGraph.aesthetics.includes(w)) || userTokens[0] || "that parameter";
        
        const openers = [
            `I have processed your query regarding ${subject}. `,
            `Analyzing the parameters for ${subject}... `,
            `My cognitive engine has evaluated your input on ${subject}. `,
            `Looking at the structural requirements for ${subject}, `
        ];

        const reasoning = [
            `As an AI architect, I interpret this through a structural lens. `,
            `I evaluate concepts like this based on how they translate into actual code. `,
            `My logic algorithms indicate this would require specific layout planning. `,
            `I am optimizing my response based on standard modern web development practices. `
        ];

        const conclusions = [
            `If you'd like, I can generate a custom HTML, CSS, and JS layout for you based on this concept.`,
            `Please specify what kind of UI component you would like me to build around this idea.`,
            `My primary directive is to write clean code—how can I architect this for you today?`,
            `Let me know if you need me to construct a specific web interface related to this.`
        ];

        const getRand = (arr) => arr[Math.floor(Math.random() * arr.length)];
        
        let constructedResponse = getRand(openers) + getRand(reasoning) + getRand(conclusions);

        // Inject dynamic concept awareness into the string
        if (userConcepts.length > 0) {
             const conceptNames = userConcepts.map(c => c.replace('target_', '')).join(' and ');
             constructedResponse = `I noticed you mentioned concepts related to **${conceptNames}**. ` + constructedResponse;
        }

        return constructedResponse;
    }
};

/**
 * End of VilotNI 1 Core Engine.
 * Execution ready for external App Interface integration.
 */