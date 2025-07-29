import fs from "fs";
import path from "path";
import { promisify } from "util";
import readline from "readline";
import axios from "axios";
import { config } from "dotenv";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { BUSINESS_IDEAS } from "./data/ideas";

config(); // Load environment variables

// Promisify file system operations
const mkdir = promisify(fs.mkdir);
const writeFile = promisify(fs.writeFile);
const readFile = promisify(fs.readFile);
const exists = promisify(fs.exists);

// LLM Provider Configuration
type LLMProvider = 'openai' | 'anthropic' | 'gemini';

interface LLMConfig {
	provider: LLMProvider;
	model: string;
	apiKey: string;
}

// Initialize LLM clients
const openaiClient = process.env.OPENAI_API_KEY ? new OpenAI({
	apiKey: process.env.OPENAI_API_KEY
}) : null;

const anthropicClient = process.env.ANTHROPIC_API_KEY ? new Anthropic({
	apiKey: process.env.ANTHROPIC_API_KEY
}) : null;

const geminiClient = process.env.GEMINI_API_KEY ? new GoogleGenerativeAI(
	process.env.GEMINI_API_KEY
) : null;

// Available providers and their models
const AVAILABLE_PROVIDERS = {
	openai: {
		name: 'OpenAI',
		models: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
		requiredEnv: 'OPENAI_API_KEY',
		client: openaiClient
	},
	anthropic: {
		name: 'Anthropic Claude',
		models: ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307', 'claude-3-opus-20240229'],
		requiredEnv: 'ANTHROPIC_API_KEY',
		client: anthropicClient
	},
	gemini: {
		name: 'Google Gemini',
		models: ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro'],
		requiredEnv: 'GEMINI_API_KEY',
		client: geminiClient
	}
} as const;

// Business ideas data structure

// Interface for the ebook structure
interface Ebook {
	title: string;
	category: string;
	plan: string;
	chapters: {
		title: string;
		content: string;
		illustrationPrompt: string;
		illustrationPath?: string;
	}[];
	markdownContent?: string;
}

// CLI interface
const rl = readline.createInterface({
	input: process.stdin,
	output: process.stdout,
});

// Helper functions
const question = (query: string): Promise<string> => {
	return new Promise((resolve) => rl.question(query, resolve));
};

const slugify = (text: string): string => {
	return text
		.toString()
		.toLowerCase()
		.replace(/\s+/g, "-") // Replace spaces with -
		.replace(/[^\w\-]+/g, "") // Remove all non-word chars
		.replace(/\-\-+/g, "-") // Replace multiple - with single -
		.replace(/^-+/, "") // Trim - from start of text
		.replace(/-+$/, ""); // Trim - from end of text
};

const generatePlanPrompt = (title: string): string => {
	return `
  Tu es un expert en entrepreneuriat à petit budget, spécialisé pour le marché ivoirien. 
  Je veux créer un ebook intitulé : "${title}"
  
  Génère un plan détaillé (8 à 12 chapitres) pour ce livre, avec des sous-sections pratiques.
  Pour chaque chapitre, suggère une idée d'illustration (ex : infographie, photo, schéma).
  
  Structure type :
  - Introduction
  - Chapitre 1 : Étude de marché
  - Chapitre 2 : Budget et financement
  - ... (autres chapitres pertinents)
  - Conclusion
  
  Retourne uniquement le plan au format Markdown, avec pour chaque chapitre :
  ### [Titre du chapitre]
  [Description du contenu]
  Illustration : [Description de l'illustration]
  `;
};

const generateIllustrationPrompt = (chapterTitle: string, ebookTitle: string): string => {
	return `
  Crée une illustration pour le chapitre "${chapterTitle}" d'un ebook sur "${ebookTitle}".
  
  Style souhaité :
  - Type : Infographie colorée / Photo réaliste
  - Éléments à inclure : Représentation visuelle du concept du chapitre
  - Couleurs vives (orange, vert, jaune)
  - Texte minimal (juste un titre si nécessaire)
  
  Décris précisément l'image à générer en 1-2 phrases.
  `;
};

const generateChapterPrompt = (chapterTitle: string, ebookTitle: string): string => {
	return `
  Rédige le chapitre "${chapterTitle}" pour un ebook intitulé "${ebookTitle}".
  
  Instructions :
  1. Longueur : 500 à 800 mots
  2. Structure :
     - Introduction (2-3 phrases)
     - Partie 1 : [Sous-thème 1] (étapes, exemples)
     - Partie 2 : [Sous-thème 2] (liste, conseils)
     - Encadré "Astuce" (1 conseil clé)
     - Encadré "Erreur à éviter" (1 piège courant)
     - Résumé (3 points clés)
  
  3. Ton :
     - Simple et direct
     - Utiliser "vous" pour impliquer le lecteur
     - Donner des exemples concrets avec estimations de prix
     - Adapter au marché ivoirien
  
  Retourne uniquement le contenu du chapitre, sans le titre.
  `;
};

// LLM API calls
const callLLM = async (prompt: string, config: LLMConfig): Promise<string> => {
	try {
		switch (config.provider) {
			case 'openai':
				if (!openaiClient) throw new Error('OpenAI client not initialized. Check OPENAI_API_KEY.');
				const openaiResponse = await openaiClient.chat.completions.create({
					model: config.model,
					messages: [{ role: "user", content: prompt }],
					temperature: 0.7,
				});
				return openaiResponse.choices[0].message.content?.trim() || '';

			case 'anthropic':
				if (!anthropicClient) throw new Error('Anthropic client not initialized. Check ANTHROPIC_API_KEY.');
				const anthropicResponse = await anthropicClient.messages.create({
					model: config.model,
					max_tokens: 4000,
					messages: [{ role: "user", content: prompt }],
					temperature: 0.7,
				});
				const content = anthropicResponse.content[0];
				return content.type === 'text' ? content.text.trim() : '';

			case 'gemini':
				if (!geminiClient) throw new Error('Gemini client not initialized. Check GEMINI_API_KEY.');
				const geminiModel = geminiClient.getGenerativeModel({ model: config.model });
				const geminiResponse = await geminiModel.generateContent(prompt);
				return geminiResponse.response.text().trim();

			default:
				throw new Error(`Unsupported provider: ${config.provider}`);
		}
	} catch (error) {
		console.error(`Error calling ${config.provider} API:`, error);
		throw error;
	}
};

const generateImage = async (prompt: string): Promise<string> => {
	try {
		// Currently only OpenAI DALL-E is supported for image generation
		if (!openaiClient) {
			throw new Error('OpenAI client not initialized. Image generation requires OPENAI_API_KEY.');
		}

		const response = await openaiClient.images.generate({
			prompt,
			n: 1,
			size: "512x512",
		});

		const imageUrl = response.data?.[0]?.url;
		if (!imageUrl) throw new Error('No image URL returned from OpenAI');

		// Download and save the image
		const imageResponse = await axios.get(imageUrl, { responseType: "arraybuffer" });
		return imageResponse.data;
	} catch (error) {
		console.error("Error generating image:", error);
		throw error;
	}
};

// Status checking functions
const getIdeaStatus = async (title: string, category: string) => {
	const slug = slugify(title);
	const categorySlug = slugify(category);
	const outputDir = path.join("output", categorySlug, slug);
	
	// Check if output directory exists
	if (!(await exists(outputDir))) {
		return { status: 'not_started', chapters: 0, images: 0, totalChapters: 0 };
	}
	
	// Check for plan file and get expected chapter count
	const planPath = path.join(outputDir, "1-title-chapters-plan.txt");
	let totalChapters = 0;
	
	if (await exists(planPath)) {
		try {
			const planContent = await readFile(planPath, 'utf-8');
			const chapterMatches = planContent.matchAll(/### (.+?)\n/g);
			totalChapters = Array.from(chapterMatches).length;
		} catch (error) {
			// If we can't read the plan, assume 0 chapters
			totalChapters = 0;
		}
	}
	
	if (totalChapters === 0) {
		return { status: 'plan_only', chapters: 0, images: 0, totalChapters: 0 };
	}
	
	// Count existing chapter files and images
	let chapters = 0;
	let images = 0;
	
	try {
		// List directory contents and count matching files
		const readdirAsync = promisify(fs.readdir);
		const dirContents = await readdirAsync(outputDir);
		
		// Count chapter text files (pattern: number-*.txt, excluding plan file)
		chapters = dirContents.filter(file => 
			file.match(/^\d+-.*\.txt$/) && file !== "1-title-chapters-plan.txt"
		).length;
		
		// Count image files (pattern: number-*.png)
		images = dirContents.filter(file => 
			file.match(/^\d+-.*\.png$/)
		).length;
		
		// Check for final markdown file
		const markdownFile = dirContents.find(file => file.endsWith('.md'));
		const isCompleted = markdownFile && chapters === totalChapters && images === totalChapters;
		
		if (isCompleted) {
			return { status: 'completed', chapters, images, totalChapters };
		} else if (chapters > 0 || images > 0) {
			return { status: 'in_progress', chapters, images, totalChapters };
		} else {
			return { status: 'plan_only', chapters, images, totalChapters };
		}
		
	} catch (error) {
		// If we can't read the directory, assume plan only if plan exists
		if (await exists(planPath)) {
			return { status: 'plan_only', chapters: 0, images: 0, totalChapters };
		}
		return { status: 'error', chapters: 0, images: 0, totalChapters: 0 };
	}
};

const formatIdeaStatus = (idea: string, status: any) => {
	switch (status.status) {
		case 'completed':
			return `${idea} ✅ [COMPLETED]`;
		case 'in_progress':
			return `${idea} 🔄 [${status.images}/${status.totalChapters}img ${status.chapters}/${status.totalChapters}chapt]`;
		case 'plan_only':
			return `${idea} 📝 [PLAN READY]`;
		case 'not_started':
			return `${idea}`;
		case 'error':
			return `${idea} ❌ [ERROR]`;
		default:
			return idea;
	}
};

// Helper function to select LLM provider and model
const selectLLMProvider = async (): Promise<LLMConfig> => {
	// Check available providers
	const availableProviders = Object.entries(AVAILABLE_PROVIDERS).filter(
		([_, config]) => config.client !== null
	);

	if (availableProviders.length === 0) {
		throw new Error('No LLM providers available. Please set at least one API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY)');
	}

	console.log('\n🤖 Available LLM Providers:');
	availableProviders.forEach(([key, config], i) => {
		console.log(`${i + 1}. ${config.name}`);
	});

	const providerChoice = await question('\nSelect LLM provider (number): ');
	const selectedProvider = availableProviders[parseInt(providerChoice) - 1];

	if (!selectedProvider) {
		throw new Error('Invalid provider selection');
	}

	const [providerKey, providerConfig] = selectedProvider;

	// Select model
	console.log(`\n📝 Available models for ${providerConfig.name}:`);
	providerConfig.models.forEach((model, i) => {
		console.log(`${i + 1}. ${model}`);
	});

	const modelChoice = await question('\nSelect model (number): ');
	const selectedModel = providerConfig.models[parseInt(modelChoice) - 1];

	if (!selectedModel) {
		throw new Error('Invalid model selection');
	}

	return {
		provider: providerKey as LLMProvider,
		model: selectedModel,
		apiKey: process.env[providerConfig.requiredEnv] || ''
	};
};

// Main ebook generation function
const generateEbook = async (title: string, category: string, llmConfig: LLMConfig, dryRun: boolean = false) => {
	const slug = slugify(title);
	const categorySlug = slugify(category);
	const outputDir = path.join("output", categorySlug, slug);

	// Create output directory (skip in dry run mode)
	if (!dryRun && !(await exists(outputDir))) {
		await mkdir(outputDir, { recursive: true });
	}

	const ebook: Ebook = {
		title,
		category,
		plan: "",
		chapters: [],
	};

	console.log(`\n🚀 Starting ${dryRun ? 'DRY RUN' : 'generation'} for: ${title}`);
	if (dryRun) {
		console.log("🔍 DRY RUN MODE: No API calls will be made, no files will be created");
	}

	// Step 1: Generate plan
	const planPath = path.join(outputDir, "1-title-chapters-plan.txt");
	if (await exists(planPath) && !dryRun) {
		console.log("↩️ Found existing plan, loading...");
		ebook.plan = await readFile(planPath, "utf-8");
	} else {
		if (dryRun) {
			console.log("📝 [DRY RUN] Would generate plan with prompt:");
			console.log("─".repeat(60));
			console.log(generatePlanPrompt(title));
			console.log("─".repeat(60));
			// Mock plan for dry run
			ebook.plan = `### Introduction
Ce chapitre présente les bases du business
Illustration : Infographie présentant les concepts clés

### Chapitre 1: Étude de marché
Analyse du marché et de la concurrence
Illustration : Graphique de l'analyse de marché

### Chapitre 2: Budget et financement
Calcul des coûts et sources de financement
Illustration : Tableau des coûts et revenus

### Chapitre 3: Mise en œuvre
Les étapes pratiques de lancement
Illustration : Timeline de mise en œuvre

### Conclusion
Résumé et prochaines étapes
Illustration : Checklist de réussite`;
			console.log("✅ [DRY RUN] Plan preview generated");
		} else {
			console.log("📝 Generating plan...");
			ebook.plan = await callLLM(generatePlanPrompt(title), llmConfig);
			await writeFile(planPath, ebook.plan);
			console.log("✅ Plan generated and saved");
		}
	}

	// Parse chapters from plan
	const chapterMatches = ebook.plan.matchAll(/### (.+?)\n([\s\S]+?)(?=###|$)/g);
	for (const match of chapterMatches) {
		const chapterTitle = match[1].trim();
		const chapterDescription = match[2].trim();

		// Extract illustration prompt from plan
		const illustrationMatch = chapterDescription.match(/Illustration : (.+)/);
		let illustrationPrompt = '';
		if (illustrationMatch) {
			illustrationPrompt = illustrationMatch[1].trim();
		} else if (dryRun) {
			illustrationPrompt = `[DRY RUN] Would generate illustration prompt for: ${chapterTitle}`;
		} else {
			illustrationPrompt = await callLLM(generateIllustrationPrompt(chapterTitle, title), llmConfig);
		}

		ebook.chapters.push({
			title: chapterTitle,
			content: "",
			illustrationPrompt,
		});
	}

	// Step 2: Generate illustrations and chapter contents
	for (let i = 0; i < ebook.chapters.length; i++) {
		const chapter = ebook.chapters[i];
		const chapterSlug = slugify(chapter.title);
		const chapterPrefix = `${i + 1}-${chapterSlug}`;

		console.log(`\n📖 Chapter ${i + 1}/${ebook.chapters.length}: ${chapter.title}`);

		// Generate illustration
		const illustrationPath = path.join(outputDir, `${chapterPrefix}.png`);
		if (dryRun) {
			console.log("🎨 [DRY RUN] Would generate illustration with prompt:");
			console.log(`   "${chapter.illustrationPrompt}"`);
			console.log(`   → Output: ${illustrationPath}`);
			chapter.illustrationPath = illustrationPath;
		} else if (await exists(illustrationPath)) {
			console.log("↩️ Found existing illustration, skipping...");
			chapter.illustrationPath = illustrationPath;
		} else {
			console.log("🎨 Generating illustration...");
			const imageData = await generateImage(chapter.illustrationPrompt);
			await writeFile(illustrationPath, imageData);
			chapter.illustrationPath = illustrationPath;
			console.log("✅ Illustration generated and saved");
		}

		// Generate chapter content
		const chapterPath = path.join(outputDir, `${chapterPrefix}.txt`);
		if (dryRun) {
			console.log("✍️ [DRY RUN] Would generate chapter content with prompt:");
			console.log("─".repeat(40));
			console.log(generateChapterPrompt(chapter.title, title));
			console.log("─".repeat(40));
			chapter.content = `[DRY RUN] Mock content for chapter: ${chapter.title}

Ce chapitre aurait été généré avec ${llmConfig.provider} (${llmConfig.model}).

Le contenu inclurait:
- Introduction au sujet
- Étapes pratiques détaillées  
- Exemples concrets avec prix
- Conseils et astuces
- Erreurs à éviter
- Résumé des points clés

Longueur estimée: 500-800 mots`;
			console.log(`   → Output: ${chapterPath}`);
		} else if (await exists(chapterPath)) {
			console.log("↩️ Found existing chapter content, loading...");
			chapter.content = await readFile(chapterPath, "utf-8");
		} else {
			console.log("✍️ Generating chapter content...");
			chapter.content = await callLLM(generateChapterPrompt(chapter.title, title), llmConfig);
			await writeFile(chapterPath, chapter.content);
			console.log("✅ Chapter content generated and saved");
		}

		// Pause point for verification (skip in dry run mode)
		if (!dryRun && i < ebook.chapters.length - 1) {
			const answer = await question("Continue to next chapter? (y/n) ");
			if (answer.toLowerCase() !== "y") {
				console.log("⏸️ Generation paused. You can resume later.");
				return;
			}
		}
	}

	// Step 3: Compile markdown
	if (dryRun) {
		console.log("\n📚 [DRY RUN] Would compile markdown...");
		console.log(`   → Output: ${path.join(outputDir, `${slug}.md`)}`);
		console.log("\n📋 [DRY RUN] Summary of what would be generated:");
		console.log(`   📁 Directory: ${outputDir}`);
		console.log(`   📝 Plan file: 1-title-chapters-plan.txt`);
		ebook.chapters.forEach((chapter, i) => {
			const chapterSlug = slugify(chapter.title);
			console.log(`   📖 Chapter ${i + 1}: ${i + 1}-${chapterSlug}.txt`);
			console.log(`   🎨 Image ${i + 1}: ${i + 1}-${chapterSlug}.png`);
		});
		console.log(`   📚 Final ebook: ${slug}.md`);
		
		console.log("\n💰 Estimated API usage:");
		const totalPrompts = 1 + ebook.chapters.length * 2; // plan + (content + illustration) per chapter
		console.log(`   📊 Total prompts: ${totalPrompts}`);
		console.log(`   🤖 Provider: ${AVAILABLE_PROVIDERS[llmConfig.provider].name}`);
		console.log(`   🧠 Model: ${llmConfig.model}`);
		console.log(`   🎨 Images: ${ebook.chapters.length} (DALL-E required)`);
	} else {
		console.log("\n📚 Compiling markdown...");
		let markdown = `# ${title}\n\n`;

		// Add introduction
		markdown += `## Introduction\n\n`;
		markdown += `Ce guide vous montrera comment ${title.toLowerCase()}.\n\n`;

		// Add chapters
		for (let i = 0; i < ebook.chapters.length; i++) {
			const chapter = ebook.chapters[i];
			const chapterSlug = slugify(chapter.title);
			const imagePath = `${i + 1}-${chapterSlug}.png`;

			markdown += `## Chapitre ${i + 1}: ${chapter.title}\n\n`;
			markdown += `![Illustration pour ${chapter.title}](${imagePath})\n\n`;
			markdown += `${chapter.content}\n\n`;
		}

		// Add conclusion
		markdown += `## Conclusion\n\n`;
		markdown += `Vous avez maintenant toutes les clés pour ${title.toLowerCase()}.\n\n`;

		// Save markdown
		const markdownPath = path.join(outputDir, `${slug}.md`);
		await writeFile(markdownPath, markdown);
		console.log(`✅ Markdown compiled and saved to ${markdownPath}`);
	}

	return ebook;
};

// Main CLI function
const main = async () => {
	console.log("📘 Business Idea Ebook Generator CLI\n");

	// Show overall progress statistics (optional)
	const showOverview = await question("📊 Show project overview? (y/n, default: y): ");
	
	if (showOverview.toLowerCase() !== 'n') {
		console.log("🔄 Scanning existing projects...");
		let totalIdeas = 0;
		let completedIdeas = 0;
		let inProgressIdeas = 0;

		for (const [category, ideas] of Object.entries(BUSINESS_IDEAS)) {
			for (const idea of ideas) {
				totalIdeas++;
				const status = await getIdeaStatus(idea, category);
				if (status.status === "completed") {
					completedIdeas++;
				} else if (status.status === "in_progress" || status.status === "plan_only") {
					inProgressIdeas++;
				}
			}
		}

		console.log("\n📊 Project Overview:");
		console.log(`   📚 Total Ideas: ${totalIdeas}`);
		console.log(`   ✅ Completed: ${completedIdeas}`);
		console.log(`   🔄 In Progress: ${inProgressIdeas}`);
		console.log(`   🆕 Not Started: ${totalIdeas - completedIdeas - inProgressIdeas}`);
	}

	// List categories
	console.log("\n📂 Categories:");
	const categories = Object.keys(BUSINESS_IDEAS);
	categories.forEach((cat, i) => {
		console.log(`${i + 1}. ${cat}`);
	});

	// Select category
	const catChoice = await question("\nSelect a category (number): ");
	const selectedCategory = categories[parseInt(catChoice) - 1];

	if (!selectedCategory) {
		console.error("Invalid category selection");
		process.exit(1);
	}

	// List ideas in category with status
	console.log(`\n💡 Ideas in ${selectedCategory}:`);
	console.log('🔄 Checking status...');
	
	const ideas = BUSINESS_IDEAS[selectedCategory as keyof typeof BUSINESS_IDEAS];
	const ideaStatuses = await Promise.all(
		ideas.map(idea => getIdeaStatus(idea, selectedCategory))
	);
	
	// Show options to filter
	console.log('\n📋 Display options:');
	console.log('1. Show all ideas');
	console.log('2. Show only incomplete ideas');
	console.log('3. Show only completed ideas');
	
	const displayChoice = await question('\nSelect display option (number, default: 1): ');
	const showCompleted = displayChoice !== '2';
	const showIncomplete = displayChoice !== '3';
	
	console.log(`\n💡 Ideas in ${selectedCategory}:`);
	
	let displayIndex = 1;
	const ideaChoiceMap: { [key: number]: number } = {};
	
	ideas.forEach((idea: string, i: number) => {
		const status = ideaStatuses[i];
		const isCompleted = status.status === 'completed';
		
		// Filter based on user choice
		if ((isCompleted && showCompleted) || (!isCompleted && showIncomplete)) {
			const formattedIdea = formatIdeaStatus(idea, status);
			console.log(`${displayIndex}. ${formattedIdea}`);
			ideaChoiceMap[displayIndex] = i; // Map display index to original index
			displayIndex++;
		}
	});
	
	if (displayIndex === 1) {
		console.log('No ideas match the selected filter.');
		return;
	}

	// Select idea
	const ideaChoice = await question("\nSelect an idea (number) or enter custom title: ");
	let selectedTitle = "";

	if (isNaN(parseInt(ideaChoice))) {
		selectedTitle = ideaChoice;
	} else {
		const choiceNum = parseInt(ideaChoice);
		const originalIndex = ideaChoiceMap[choiceNum];
		
		if (originalIndex !== undefined) {
			selectedTitle = ideas[originalIndex];
		} else {
			console.error("Invalid idea selection");
			process.exit(1);
		}
	}

	if (!selectedTitle) {
		console.error("Invalid idea selection");
		process.exit(1);
	}
	
	// Show selected idea status
	const selectedIdeaStatus = await getIdeaStatus(selectedTitle, selectedCategory);
	if (selectedIdeaStatus.status === 'completed') {
		console.log(`\n⚠️  Selected idea is already completed!`);
		const continueChoice = await question('Continue anyway? (y/n): ');
		if (continueChoice.toLowerCase() !== 'y') {
			console.log('👋 Exiting...');
			process.exit(0);
		}
	} else if (selectedIdeaStatus.status === 'in_progress') {
		console.log(`\n📊 Selected idea is partially completed:`);
		console.log(`   📝 Chapters: ${selectedIdeaStatus.chapters}/${selectedIdeaStatus.totalChapters}`);
		console.log(`   🎨 Images: ${selectedIdeaStatus.images}/${selectedIdeaStatus.totalChapters}`);
		console.log(`\n💡 Generation will continue from where it left off.`);
	}

	// Select LLM provider and model
	let llmConfig: LLMConfig;
	try {
		llmConfig = await selectLLMProvider();
		console.log(`\n✅ Selected: ${AVAILABLE_PROVIDERS[llmConfig.provider].name} - ${llmConfig.model}`);
	} catch (error) {
		console.error("\n❌ Error selecting LLM provider:", error);
		rl.close();
		return;
	}

	// Ask for dry run mode
	console.log('\n🔍 Generation Mode:');
	console.log('1. Normal generation (uses API credits)');
	console.log('2. Dry run (preview only, no API calls)');
	
	const modeChoice = await question('\nSelect mode (number): ');
	const isDryRun = modeChoice === '2';
	
	if (isDryRun) {
		console.log('\n🔍 DRY RUN MODE SELECTED');
		console.log('   • No API calls will be made');
		console.log('   • No files will be created');
		console.log('   • Preview prompts and structure only');
	}

	// Start generation
	try {
		await generateEbook(selectedTitle, selectedCategory, llmConfig, isDryRun);
		if (isDryRun) {
			console.log("\n🔍 Dry run completed! Review the preview above.");
			console.log("💡 To generate the actual ebook, run again and select normal mode.");
		} else {
			console.log("\n🎉 Ebook generation completed successfully!");
			
			// Show final status and file location
			const finalStatus = await getIdeaStatus(selectedTitle, selectedCategory);
			const slug = slugify(selectedTitle);
			const categorySlug = slugify(selectedCategory);
			const outputDir = path.join("output", categorySlug, slug);
			
			console.log("\n📊 Generation Summary:");
			console.log(`   📁 Output Directory: ${outputDir}`);
			console.log(`   📝 Chapters Generated: ${finalStatus.chapters}/${finalStatus.totalChapters}`);
			console.log(`   🎨 Images Generated: ${finalStatus.images}/${finalStatus.totalChapters}`);
			console.log(`   📚 Status: ${finalStatus.status === 'completed' ? '✅ COMPLETED' : '🔄 IN PROGRESS'}`);
			
			if (finalStatus.status === 'completed') {
				console.log(`\n🎊 Your ebook "${selectedTitle}" is ready!`);
				console.log(`📖 Open ${path.join(outputDir, `${slug}.md`)} to view the complete ebook.`);
			}
		}
	} catch (error) {
		console.error("\n❌ Error during ebook generation:", error);
	} finally {
		rl.close();
	}
};

// Run the program
main().catch(console.error);
