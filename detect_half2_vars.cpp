#include <sstream>
#include <string>
#include <vector> //for stack + info storage
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdio.h> //for getting file fd
#include "clang/Lex/Lexer.h" //for getting source code from the AST.
#include "clang/Basic/SourceLocation.h"
#include "clang/AST/AST.h"
#include "clang/AST/ParentMap.h"//for getting parent of a node
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_MY_PARSER true


using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;
using namespace std;
static llvm::cl::OptionCategory ToolingSampleCategory("Tooling Sample");

  vector<pair<string,string> > IndexVars; //variables used for thread index
  vector<pair<string,string> > NoThreadVars; //possible var used for identifying number of threads in device code: if (idx< n) {do sthing} else {do nothing}
  vector<pair<string,string> > half2Vars; // variable are in half2 precision. due to complication of conversion from floating point literal to half2. there may be some constant/var decl need to be in float.




// By implementing RecursiveASTVisitor, we can specify which AST nodes
// we're interested in by overriding relevant methods.
class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor> {
public:
  MyASTVisitor(Rewriter &R,ASTContext *Context) : TheRewriter(R), Context(Context), CurrentFunc(nullptr), CudaCode(false){}

	bool IsFloatingPointType(string typeStr){
		return (typeStr.find("double") != string::npos) || (typeStr.find("float") != string::npos);
	}
	bool IsIntegerType(string typeStr){
		return (typeStr.find("int") != string::npos) || (typeStr.find("long") != string::npos);
	}
	bool IsThreadIDRelated(string expr){
		return (expr.find("blockIdx") != string::npos) || (expr.find("blockDim") != string::npos)|| (expr.find("threadIdx") != string::npos);
	}

  
  bool VisitParmVarDecl(ParmVarDecl *Decl)
  {
	  cout<<"visit ParamDecl \n";
	  cout<<(Decl->getOriginalType()).getAsString()<<"\n";
	  cout<<Decl->getNameAsString()<<"\n";
	  string typeStr = (Decl->getOriginalType()).getAsString(); //currently match str with double/float. don't consider whether it's pointer or sth
	  //Decl->dump();
	  if(CudaCode)
	  {
		  if (IsFloatingPointType(typeStr) && Decl->getNameAsString().length()!=0 )
		  {
			  //TODO: add declName to a list to retrieve later.
			  //~ int dimension = count(typeStr.begin(), typeStr.end(), '*'); //dirty hack for pointer type. need to read Clang Docs later
			  //~ //replace double * with __half2 *
			  //~ //TheRewriter.ReplaceText(Decl->getTypeSpecStartLoc(), (Decl->getOriginalType()).getAsString().size(), "__half2 " + string(dimension, '*'));
				//~ if (CurrentFunc!= nullptr)
					//~ half2Vars.push_back( make_pair(Decl->getNameAsString(),CurrentFunc->getNameAsString()));
				//~ else
					//~ half2Vars.push_back( make_pair(Decl->getNameAsString(),""));
					
		  } else if (IsIntegerType(typeStr)){
			  if (CurrentFunc!=nullptr)
				NoThreadVars.push_back( make_pair(Decl->getNameAsString(), CurrentFunc->getNameAsString()));
				else //never happens ??
				NoThreadVars.push_back( make_pair(Decl->getNameAsString(), ""));
		  }
	 }
	
	  return true;
  }
	bool VisitVarDecl(VarDecl* Decl){
	  cout<<"visit VarDecl \n";
	  cout<<(Decl->getType()).getAsString()<<"\n";
	  
	  cout<<Decl->getNameAsString()<<"\n";	
	  	 //~ cout<<Decl->getNameAsString().length()<<"\n";
	  cout<<"has init "<< Decl->hasInit()<<"\n";
	  string typeStr = (Decl->getType()).getAsString(); 
	  if(Decl->hasAttr<CUDAConstantAttr>() && IsFloatingPointType(typeStr) && Decl->getNameAsString().length()!=0)//global variable in cuda must go with __constant__ atttribute 
			half2Vars.push_back( make_pair(Decl->getNameAsString(),""));
	  if(CudaCode)
	  {
		  if (IsFloatingPointType(typeStr)){
			 ; //handle half2 decl
			 	if (CurrentFunc!= nullptr && Decl->getNameAsString().length()!=0)
					half2Vars.push_back( make_pair(Decl->getNameAsString(),CurrentFunc->getNameAsString()));
				else{ 
					;
				}
		  } else if (IsIntegerType(typeStr)){//detect variables used for thread index
			 if (Decl->hasInit()){
				 Expr* InitStmt = Decl->getInit();
				 string InitSrc = Lexer::getSourceText(CharSourceRange::getCharRange(InitStmt->getSourceRange()),Context->getSourceManager(),LangOptions(), 0);
				 if(DEBUG_MY_PARSER)
					cout<<"init "<<InitSrc<<"\n";
				if (IsThreadIDRelated(InitSrc)){
					if (CurrentFunc!=nullptr)
						IndexVars.push_back( make_pair(Decl->getNameAsString(),CurrentFunc->getNameAsString()));
					else //never happens ??
						IndexVars.push_back( make_pair(Decl->getNameAsString(),""));
					}
			 }
		  }
		  
	  }	  	
	return true;
	}


  bool VisitFunctionDecl(FunctionDecl *f) {
    // Only function definitions (with bodies), not declarations. //will care header files later
    
	if (f->hasAttr<CUDAGlobalAttr>() || f->hasAttr<CUDADeviceAttr>()){ //note down __device__ float tetstfunc to convert  to __half2 return type
		if (IsFloatingPointType(f->getReturnType().getAsString())){
				half2Vars.push_back( make_pair(f->getNameInfo().getAsString (),f->getNameInfo().getName().getAsString ()));
			
			}
		CudaCode = true;
		
		}
	else 
		CudaCode = false;

    if (f->hasBody()) {
		CurrentFunc = f;
      Stmt *FuncBody = f->getBody();

      // Type name as string
      QualType QT = f->getReturnType();
      std::string TypeStr = QT.getAsString();

      // Function name
      DeclarationName DeclName = f->getNameInfo().getName();
      std::string FuncName = DeclName.getAsString();
		//convert __device__ func returns float => half2
		//note down the potential funcs to convert;
    }
    
    return true;
  }
  
  bool VisitCUDADeviceAttr(CUDADeviceAttr* Attr){
	  if (CurrentFunc!= nullptr)
		cout <<"visit cuda Dev attr end of func "<<CurrentFunc->getNameAsString () <<" \n";
	  CurrentFunc = nullptr;
	  CudaCode = false;
	  return true;
	  }

	bool VisitCUDAGlobalAttr(CUDAGlobalAttr* Attr){
		if (CurrentFunc!= nullptr)
			cout <<"visit cuda Global Attr end of func "<<CurrentFunc->getNameAsString () <<"\n";
		CurrentFunc = nullptr;
		CudaCode = false;
		return true;
		}

private:
  Rewriter &TheRewriter;
  bool CudaCode; //detect whether we are parsing cuda code or host code
  ASTContext *Context;
  FunctionDecl* CurrentFunc; //not a good way to get current func, will change later
  //rule to store vars: pair them with their parent function varname*****function_name. global vars => pair *****global    

};


// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser.

class MyASTConsumer : public ASTConsumer {
public:
  MyASTConsumer(Rewriter &R, ASTContext *Context) : Visitor(R,Context) {}


	virtual void HandleTranslationUnit(ASTContext &Context) {
	
		Visitor.TraverseDecl(Context.getTranslationUnitDecl());
	}


private:
  MyASTVisitor Visitor;

};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  
  void writeHalf2VarsToFile(string fileName){
	ofstream myFile;
	myFile.open (fileName);
	for(int i = 0; i < half2Vars.size(); i++) {
		myFile <<half2Vars[i].first << " " <<(half2Vars[i].second==""?"null":half2Vars[i].second) <<"\n";
	}
	myFile.close();
		
	}

  
  void EndSourceFileAction() override {
    SourceManager &SM = TheRewriter.getSourceMgr();
    llvm::errs() << "** EndSourceFileAction for: "
                 << SM.getFileEntryForID(SM.getMainFileID())->getName() << "\n";

    // Now emit the rewritten buffer.
  //  TheRewriter.getEditBuffer(SM.getMainFileID()).write(llvm::outs());
	writeHalf2VarsToFile("half2VarsList.txt");
  }
  

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
													 
    llvm::errs() << "** Creating AST consumer for: " << file << "\n";
    
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    
    return  llvm::make_unique<MyASTConsumer>(TheRewriter,&CI.getASTContext());

  }

private:
  Rewriter TheRewriter;
};

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, ToolingSampleCategory);
  //CommonOptionsParser op(argc, argv);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());

  // ClangTool::run accepts a FrontendActionFactory, which is then used to
  // create new objects implementing the FrontendAction interface. Here we use
  // the helper newFrontendActionFactory to create a default factory that will
  // return a new MyFrontendAction object every time.
  // To further customize this, we could create our own factory class.
  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}

